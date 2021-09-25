import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Literal

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

# Speech synthesis
import pyttsx3

# Import WebRTC plugin from streamlit
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

USERS = {'bob': hash('strawberry')}

logger = logging.getLogger(__name__)

# Download file using `urllib`
def download_file(url, download_to: Path, expected_size=None):
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEBIBYTES = 2.0 ** 20.0
                PACKETLENGTH = 8192
                while True:
                    data = response.read(PACKETLENGTH) # Read packet
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MiB)"
                        % (url, counter / MEBIBYTES, length / MEBIBYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Hide visuals after download
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# Default ICE servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.title("Audit panel")

    if 'login' not in st.session_state:
        st.session_state['login'] = False

    if st.session_state['login'] == False:
        username_field = st.empty()
        username = username_field.text_input('Username: ', value='')

        password_field = st.empty()
        password = password_field.text_input('Password: ', value='', type='password')

        if st.button('Login'):
            if username != '' and password != '':
                if username in USERS and USERS[username] == hash(password):
                    st.session_state['login'] = True
                    username_field.empty()
                    password_field.empty()
                else:
                    st.warning('Either username or password is incorrect')
                    st.stop()
            else:
                st.warning('Username and password must be entered')
                st.stop()

        if st.button('Register'):
            st.warning('TBD. Just you wait!')
            st.stop()

    if st.session_state['login'] == True:
        use_ext = st.checkbox('Use external file?', value=False)
        if not use_ext: 
            object_detection()
        else:
            object_detection_ext()

        # Log all living threads
        logger.debug("=== Alive threads ===")
        for thread in threading.enumerate():
            if thread.is_alive():
                logger.debug(f"  {thread.name} ({thread.ident})")

def object_detection():
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = Path(__file__).parent / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = Path(__file__).parent / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class Processor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            # Initialize detection model (proto and model)
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # NOTE: Iterate over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # NOTE: extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # NOTE: Display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        # NOTE: On recv do annotation on a video frame
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24") # NOTE: Allocate frame buffer
            blob = cv2.dnn.blobFromImage( # NOTE: Create handle for frame processing
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob) # NOTE: Bind net to the handle
            detections = self._net.forward() # NOTE: Get detections

            # NOTE: Get annotated video frame and data pairs list
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: Put results in a queue for further use
            # XXX: Current method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            # NOTE: Finally, return processed frame
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    # NOTE: Initialize WebRTC context
    webrtc_ctx = webrtc_streamer(
        key="detection", # NOTE: Any name
        mode=WebRtcMode.SENDRECV, # NOTE: Has to be `SENDRECV` to allow frame processing
        rtc_configuration=RTC_CONFIGURATION, # NOTE: Give a list of known ICE servers
        video_processor_factory=Processor, # NOTE: Set processor class (should have `recv(frame)`)
        async_processing=True, # NOTE: Async enabled
        media_stream_constraints={"video": True}, # NOTE: Only allow video stream
    )

    # NOTE: Put annotations in a UI table as well
    if webrtc_ctx.state.playing:
        # NOTE: Create empty UI object
        labels = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            if webrtc_ctx.video_processor:
                # NOTE: Receive results periodically
                try:
                    result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                # NOTE: Build a table
                labels.table(result)
            else:
                break

def object_detection_ext():
    ext_file = st.file_uploader('Choose video file')
    if ext_file is not None:
        st.write(ext_file)

    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = Path(__file__).parent / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = Path(__file__).parent / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    st.write()

    class Detection(NamedTuple):
        name: str
        prob: float

    def player():
        return MediaPlayer(str(Path(__file__).parent / f"data/{ext_file.name}"))

    class DetectionProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            # Initialize detection model (proto and model)
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # NOTE: Iterate over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # NOTE: extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # NOTE: Display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        # NOTE: On recv do annotation on a video frame
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24") # NOTE: Allocate frame buffer
            blob = cv2.dnn.blobFromImage( # NOTE: Create handle for frame processing
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob) # NOTE: Bind net to the handle
            detections = self._net.forward() # NOTE: Get detections

            # NOTE: Get annotated video frame and data pairs list
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: Put results in a queue for further use
            # XXX: Current method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            # NOTE: Finally, return processed frame
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    class Processor(VideoProcessorBase):
        def __init__(self) -> None:
            pass

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="detection-via-external",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True},
        player_factory=player,
        video_processor_factory=DetectionProcessor,
    )

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()