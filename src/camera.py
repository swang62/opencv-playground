"""Platform-native camera backends."""

from __future__ import annotations

import platform
import threading

import cv2
import numpy as np

IS_MACOS = platform.system() == "Darwin"

# ---- macOS native camera (AVFoundation via pyobjc) --------------------------

HAVE_NATIVE_CAMERA = False

if IS_MACOS:
    try:
        import AVFoundation as AV
        import CoreMedia as CM
        import CoreVideo as CV
        import Foundation as FN
        from objc import super as objc_super

        HAVE_NATIVE_CAMERA = True
    except ImportError:
        pass


if HAVE_NATIVE_CAMERA:

    class FrameReceiver(FN.NSObject):
        """Receives camera frames from AVFoundation on a serial queue.

        Implements the AVCaptureVideoDataOutputSampleBufferDelegate
        protocol. Converts CVPixelBuffer (BGRA) to a BGR numpy array.
        """

        def init(self):
            self = objc_super().init()
            self._lock = threading.Lock()
            self._current_frame: np.ndarray | None = None
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, output, sample_buffer, connection
        ):
            image_buffer = CM.CMSampleBufferGetImageBuffer(sample_buffer)
            if image_buffer is None:
                return

            CV.CVPixelBufferLockBaseAddress(image_buffer, 0)
            try:
                w = CV.CVPixelBufferGetWidth(image_buffer)
                h = CV.CVPixelBufferGetHeight(image_buffer)
                bpr = CV.CVPixelBufferGetBytesPerRow(image_buffer)
                addr = CV.CVPixelBufferGetBaseAddress(image_buffer)

                import ctypes

                raw = (ctypes.c_uint8 * (h * bpr)).from_address(addr)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpr // 4, 4)
                # BGRA -> BGR, crop to actual visible width
                frame = arr[:, :w, :3].copy()
            finally:
                CV.CVPixelBufferUnlockBaseAddress(image_buffer, 0)

            with self._lock:
                self._current_frame = frame

        def get_frame(self) -> np.ndarray | None:
            with self._lock:
                if self._current_frame is None:
                    return None
                return self._current_frame.copy()

    class NativeMacCamera:
        """macOS-native camera backed by AVFoundation.

        Avoids the ``cv2.VideoCapture`` AVFoundation backend which can
        deadlock the Window Server on macOS.
        """

        def __init__(self, camera_index: int = 0):
            self._camera_index = camera_index
            self._session: AV.AVCaptureSession | None = None
            self._receiver: FrameReceiver | None = None

        def start(self):
            self._receiver = FrameReceiver.alloc().init()

            devices = AV.AVCaptureDevice.devicesWithMediaType_("vide")
            if not devices or self._camera_index >= len(devices):
                raise RuntimeError(f"Camera {self._camera_index} not found")
            device = devices[self._camera_index]

            input_obj, error = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(
                device, None
            )
            if error is not None:
                raise RuntimeError(f"Camera input error: {error}")

            output = AV.AVCaptureVideoDataOutput.alloc().init()
            output.setVideoSettings_(
                {CV.kCVPixelBufferPixelFormatTypeKey: CV.kCVPixelFormatType_32BGRA}
            )
            queue = FN.dispatch_queue_create("camera-capture", None)
            output.setSampleBufferDelegate_queue_(self._receiver, queue)

            self._session = AV.AVCaptureSession.alloc().init()
            self._session.setSessionPreset_(AV.AVCaptureSessionPreset1280x720)
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
    if HAVE_NATIVE_CAMERA:
        cam = NativeMacCamera(camera_id)
        cam.start()
        return cam

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_id} failed to open")
    return cap
