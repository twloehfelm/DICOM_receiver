import os
import shutil
from datetime import datetime
import threading
from pathlib import Path
from pynetdicom import (
    AE, debug_logger, evt, AllStoragePresentationContexts,
    StoragePresentationContexts, ALL_TRANSFER_SYNTAXES
)
from pydicom import dcmread

# Verification class for C-ECHO (https://pydicom.github.io/pynetdicom/stable/examples/verification.html)
from pynetdicom.sop_class import Verification

import segment
from genericanatomycolors import GenericAnatomyColors

import pydicom
import numpy as np

from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg.content import SegmentDescription
from highdicom.content import PixelMeasuresSequence
from highdicom.seg.enum import (
    SegmentAlgorithmTypeValues,
    SegmentationTypeValues
)
from highdicom.seg.sop import Segmentation
from detectron2.utils.visualizer import GenericMask

ae = AE()
# Accept storage of all SOP classes
storage_sop_classes = [
    cx.abstract_syntax for cx in AllStoragePresentationContexts
]
for uid in storage_sop_classes:
    ae.add_supported_context(uid, ALL_TRANSFER_SYNTAXES)

ae.add_supported_context(Verification)

"""
dict with
  key: 'dcmstore/received/{mrn}/{accnum}'
  val: datetime of last received file
"""
last_received_time = {}

Path('dcmstore/received').mkdir(parents=True, exist_ok=True)
Path('dcmstore/queue').mkdir(parents=True, exist_ok=True)
Path('dcmstore/processed').mkdir(parents=True, exist_ok=True)

# Preload with any studies left over from prior runs
received_pts = [x for x in Path('dcmstore/received').iterdir() if x.is_dir()]
for pt in received_pts:
    studies = [x for x in pt.iterdir() if x.is_dir()]
    for s in studies:
        last_received_time[s] = datetime.now()

# recursively merge two folders including subfolders


def mergefolders(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(str(root_src_dir), str(root_dst_dir), 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)

# Implement a handler for evt.EVT_C_ECHO (https://pydicom.github.io/pynetdicom/stable/examples/verification.html)


def handle_echo(event):
    """Handle a C-ECHO request event."""
    return 0x0000


def handle_store(event, storage_dir):
    """
    Handle EVT_C_STORE events
    Saves to:
      dcmstore/
        received/
          {mrn}/
            {accnum}/
              {series num}_{series desc}/   ###  If present  ##
                {SOPInstanceUID}.dcm
    """
    ds = event.dataset
    ds.file_meta = event.file_meta
    save_loc = storage_dir/ds.PatientID/ds.AccessionNumber
    last_received_time[save_loc] = datetime.now()

    if ds.SeriesNumber is not None:
        series_desc = str(ds.SeriesNumber).zfill(2)
        if "SeriesDescription" in ds:
            series_desc += '_' + ds.SeriesDescription.replace('/', '_')
        save_loc = save_loc/series_desc

    try:
        save_loc.mkdir(parents=True, exist_ok=True)
    except:
        # Unable to create output dir, return failure status
        return 0xC001

    save_loc = save_loc/ds.SOPInstanceUID
    # Because SOPInstanceUID includes several '.' you can't just use
    #   with_suffix or else it will replaces the portion of the UID that follows
    #   the last '.' with '.dcm', truncating the actual UID
    save_loc = save_loc.with_suffix(save_loc.suffix + '.dcm')
    ds.save_as(save_loc, write_like_original=False)

    return 0x0000


def send_dcm(study):
    """
    Send all .dcm files in {study} (recursive) to the remoteAE defined in .env
    """
    files = [x for x in study.glob('**/*.dcm')]

    remoteAE = AE()
    remoteAE.requested_contexts = StoragePresentationContexts
    assoc = remoteAE.associate('staging', 4242)
    if not assoc.is_established:
        print('Could not establish SCU association')
        return None

    for f in files:
        ds = dcmread(f)
        status = assoc.send_c_store(ds)
        if status:
            # If the storage request succeeded this will be 0x0000
            print('C-STORE request status: 0x{0:04x}'.format(status.Status))
        else:
            print('Connection timed out, was aborted or received invalid response')

    assoc.release()


# List of event handlers
handlers = [
    (evt.EVT_C_STORE, handle_store, [Path('dcmstore/received')]),
    (evt.EVT_C_ECHO, handle_echo)
]

# Supposedly increases transfer speed
# ref: https://pydicom.github.io/pynetdicom/dev/examples/storage.html#storage-scp
ae.maximum_pdu_size = 0


def check_studies():
    """
    Checks q60sec for studies with no new images in >= 2 min
    Assume these stale studies have finished being sent
    Move from `received` => `queue` folder for further processing
    Remove empty dirs from `received` folder
    """
    threading.Timer(60.0, check_studies).start()
    stale_studies = [s for s in last_received_time if (
        datetime.now() - last_received_time[s]).total_seconds() >= 120]
    for old in stale_studies:
        new = 'dcmstore/queue'/old.relative_to('dcmstore/received')
        mergefolders(old, new)
        shutil.rmtree(old)
        last_received_time.pop(old)
        try:
            old.parent.rmdir()
        except OSError:
            """
            Dir not empty. Do nothing. The server may be receiving another study from
              the same patient and that study might still be in progress
            """


check_studies()

predictor, classes = segment.prepare_predictor()
algorithm_identification = AlgorithmIdentificationSequence(
    name='dicomseg',
    version='v0.1',
    family=codes.cid7162.ArtificialIntelligence
)
categories = ['left adrenal', 'left kidney', 'liver',
              'pancreas', 'right adrenal', 'right kidney', 'spleen']


def segment_study(study_dir):
    path = Path(study_dir)
    series = [x for x in path.iterdir() if x.is_dir()]
    for s in series:
        dcms = list(s.glob('*.dcm'))
        if len(dcms) > 0:
            ds = pydicom.dcmread(dcms[0])
            series_num = ds.SeriesNumber
            if "SliceThickness" in ds and "PixelSpacing" in ds:
                slice_thickness = ds.SliceThickness
                pixel_spacing = ds.PixelSpacing
                pixel_measures = PixelMeasuresSequence(
                    pixel_spacing=pixel_spacing, slice_thickness=slice_thickness, spacing_between_slices=None)
                if "ImageType" in ds and all(x in ds.ImageType for x in ["AXIAL", "ORIGINAL", "PRIMARY"]) and "SliceThickness" in ds and ds.SliceThickness >= 3:
                    image_datasets = [pydicom.dcmread(str(f)) for f in dcms]

                    mask = {}
                    for c in categories:
                        mask[c] = np.zeros(
                            shape=(
                                len(image_datasets),
                                image_datasets[0].Rows,
                                image_datasets[0].Columns
                            ),
                            dtype=bool
                        )

                    for num, ds in enumerate(image_datasets):
                        im = ds.pixel_array
                        im = im * ds.RescaleSlope + ds.RescaleIntercept
                        outputs = predictor(im)
                        predictions = outputs["instances"].to("cpu")

                        pred_labels = {x: categories[x]
                                       for x in predictions.pred_classes.tolist()}

                        for pred_label in pred_labels:
                            organ = pred_labels[pred_label]
                            preds = predictions[predictions.pred_classes == pred_label]
                            if preds.has("pred_masks"):
                                masks = np.asarray(preds.pred_masks)
                                masks = [GenericMask(
                                    x, x.shape[0], x.shape[1]) for x in masks]
                                masks = [mask.mask for mask in masks]
                                if masks and not np.all((masks == 0)):
                                    mask[organ][num] = np.maximum.reduce(masks)

                    for c in categories:
                        if c in mask and not np.all((mask[c] == False)):
                            # Describe the segment
                            description_segment_1 = SegmentDescription(
                                segment_number=1,
                                segment_label=c,
                                segmented_property_category=codes.cid7150.AnatomicalStructure,
                                segmented_property_type=codes.cid7166.Organ,
                                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                                algorithm_identification=algorithm_identification,
                                tracking_uid=generate_uid(),
                                tracking_id='dicomseg segmentation'
                            )

                            # Create the Segmentation instance
                            seg_dataset = Segmentation(
                                source_images=image_datasets,
                                pixel_array=mask[c],
                                segmentation_type=SegmentationTypeValues.BINARY,
                                segment_descriptions=[description_segment_1],
                                series_instance_uid=generate_uid(),
                                series_number=series_num,
                                pixel_measures=pixel_measures,
                                sop_instance_uid=generate_uid(),
                                instance_number=1,
                                manufacturer='UC Davis',
                                manufacturer_model_name='DICOM SEG Segmentation',
                                software_versions='v0.1',
                                device_serial_number='90210',
                            )

                            new = 'dcmstore/processed' / \
                                s.relative_to('dcmstore/queue')
                            new.mkdir(parents=True, exist_ok=True)
                            output_dir = new/'SEGS'
                            output_dir.mkdir(parents=True, exist_ok=True)
                            fn = c.replace(' ', '_')+'_seg.dcm'
                            output_path = output_dir/fn
                            seg_dataset.SeriesDescription = c
                            if c in GenericAnatomyColors:
                                dicomLab = GenericAnatomyColors[c]['DicomLab']
                            else:
                                dicomLab = GenericAnatomyColors['tissue']['DicomLab']
                            seg_dataset.SegmentSequence[0].add_new(
                                [0x0062, 0x000d], 'US', dicomLab)
                            seg_dataset.save_as(output_path)

                    try:
                        # MOVE the SERIES to the PROCESSED folder
                        mergefolders(s, new)
                    except FileNotFoundError:
                        """Do nothing"""
                    try:
                        # Remove the STUDY folder if there are no other SERIES
                        s.parent.rmdir()
                    except OSError:
                        """
                        Dir not empty. Do nothing. There are other series in the pt jacket
                        and some may be segmented too. Will clear folder from queue after
                        all series are processed.
                        """
                    # Send each SERIES and SEG to STAGING SCP
                    try:
                        send_dcm(new)
                    except UnboundLocalError:
                        """New var not set yet - do nothing"""

    shutil.rmtree(study_dir, ignore_errors=True)

    try:
        study_dir.parent.rmdir()
    except OSError:
        """
        Dir not empty. Do nothing. Patient has other exams than need to be
        segmented.
        """


def process_from_queue():
    threading.Timer(120, process_from_queue).start()
    queue_studies = [x for x in Path(
        'dcmstore/queue/').glob('*/*') if x.is_dir()]
    for study in queue_studies:
        segment_study(study)


process_from_queue()

ae.start_server(
    ('', 11112),  # Start server on localhost port 11112
    block=True,  # Socket operates in blocking mode
    ae_title=os.environ['AIRRECEIVER_AE'],
    evt_handlers=handlers
)
