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

import pydicom
import numpy as np
import boto3

from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

logsdir = Path('/app/logs')

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

    with open(logsdir/'received.log', 'a+') as f:
        f.write(str(datetime.now()) + ',' + str(ds.PatientID) + ',' +
                str(ds.AccessionNumber) + ',' + str(save_loc) + ',' + str(ds.SOPInstanceUID) + '\n')

    save_loc = save_loc/ds.SOPInstanceUID
    # Because SOPInstanceUID includes several '.' you can't just use
    #   with_suffix or else it will replaces the portion of the UID that follows
    #   the last '.' with '.dcm', truncating the actual UID
    save_loc = save_loc.with_suffix(save_loc.suffix + '.dcm')
    ds.save_as(save_loc, write_like_original=False)

    return 0x0000


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

    s3_client = boto3.client('s3')
    bucket = os.environ['s3_bucket']

    for old in stale_studies:
        if os.environ['archive_to'] == 's3':
            files = [x for x in old.rglob('*') if x.is_file()]
            for f in files:
                with open(logsdir/'s3_archived.log', 'a+') as f:
                    f.write(str(datetime.now()) + ',' +
                            str(f.relative_to('dcmstore/received')) + '\n')
                s3_client.upload_file(str(f), bucket, str(
                    f.relative_to('dcmstore/received')))
        else:
            with open(logsdir/'local_queue_archived.log', 'a+') as f:
                f.write(str(datetime.now()) + ',' +
                        str(old.relative_to('dcmstore/received')) + '\n')
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

ae.start_server(
    ('', 11112),  # Start server on localhost port 11112
    block=True,  # Socket operates in blocking mode
    ae_title=os.environ['AIRRECEIVER_AE'],
    evt_handlers=handlers
)
