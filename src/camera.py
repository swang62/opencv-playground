"""Platform-native camera backends."""

from __future__ import annotations

import platform
import threading

import cv2
import numpy as np

_IS_MACOS = platform.system() == "Darwin"

# ---- macOS native camera (AVFoundation via pyobjc) --------------------------

_HAVE_NATIVE_CAMERA = False

if _IS_MACOS:
    try:
        import AVFoundation as _AV
        import CoreMedia as _CM
        import CoreVideo as _CV
        import Foundation as _FN
        from objc import super as _objc_super

        _HAVE_NATIVE_CAMERA = True
    except ImportError:
        pass


if _HAVE_NATIVE_CAMERA:

    class _FrameReceiver(_FN.NSObject):
        """Receives camera frames from AVFoundation on a serial queue.

        Implements the AVCaptureVideoDataOutputSampleBufferDelegate
        protocol. Converts CVPixelBuffer (BGRA) to a BGR numpy array.
        """

        def init(self):
            self = _objc_super().init()
            self._lock = threading.Lock()
            self._frame: np.ndarray | None = None
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, output, sample_buffer, connection
        ):
            image_buffer = _CM.CMSampleBufferGetImageBuffer(sample_buffer)
            if image_buffer is None:
                return

            _CV.CVPixelBufferLockBaseAddress(image_buffer, 0)
            try:
                w = _CV.CVPixelBufferGetWidth(image_buffer)
                h = _CV.CVPixelBufferGetHeight(image_buffer)
                bpr = _CV.CVPixelBufferGetBytesPerRow(image_buffer)
                addr = _CV.CVPixelBufferGetBaseAddress(image_buffer)

                import ctypes

                raw = (ctypes.c_uint8 * (h * bpr)).from_address(addr)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpr // 4, 4)
                # BGRA -> BGR, crop to actual visible width
                frame = arr[:, :w, :3].copy()
            finally:
                _CV.CVPixelBufferUnlockBaseAddress(image_buffer, 0)

            with self._lock:
                self._frame = frame

        def get_frame(self) -> np.ndarray | None:
            with self._lock:
                if self._frame is None:
                    return None
                return self._frame.copy()

    class NativeMacCamera:
        """macOS-native camera backed by AVFoundation.

        Avoids the ``cv2.VideoCapture`` AVFoundation backend which can
        deadlock the Window Server on macOS.
        """

        def __init__(self, camera_id: int = 0):
            self._camera_id = camera_id
            self._session: _AV.AVCaptureSession | None = None
            self._receiver: _FrameReceiver | None = None

        def start(self):
            self._receiver = _FrameReceiver.alloc().init()

            devices = _AV.AVCaptureDevice.devicesWithMediaType_("vide")
            if not devices or self._camera_id >= len(devices):
                raise RuntimeError(f"Camera {self._camera_id} not found")
            device = devices[self._camera_id]

            input_obj, error = _AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(
                device, None
            )
            if error is not None:
                raise RuntimeError(f"Camera input error: {error}")

            output = _AV.AVCaptureVideoDataOutput.alloc().init()
            output.setVideoSettings_(
                {_CV.kCVPixelBufferPixelFormatTypeKey: _CV.kCVPixelFormatType_32BGRA}
            )
            queue = _FN.dispatch_queue_create("camera-capture", None)
            output.setSampleBufferDelegate_queue_(self._receiver, queue)

            self._session = _AV.AVCaptureSession.alloc().init()
            self._session.setSessionPreset_(_AV.AVCaptureSessionPreset1280x720)
            self._session.addInput_(input_obj)
            self._session.addOutput_(output)
            self._session.startRunning()

        def stop(self):
            if self._session is not None:
                self._session.stopRunning()
                self._session = None
            self._receiver = None

        def read(self):
            if self._receiver is None:
                return False, None
            frame = self._receiver.get_frame()
            if frame is None:
                return False, None
            return True, frame

        def release(self):
            self.stop()

        def isOpened(self) -> bool:
            return self._session is not None


# ---- Camera factory ----------------------------------------------------------


def create_camera(camera_id: int):
    """Return a camera object for the current platform.

    On macOS with pyobjc available uses NativeMacCamera to avoid the
    OpenCV AVFoundation deadlock. Falls back to cv2.VideoCapture.
    """
    if _HAVE_NATIVE_CAMERA:
        cam = NativeMacCamera(camera_id)
        cam.start()
        return cam

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_id} failed to open")
    return cap
