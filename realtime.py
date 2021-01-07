from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from age_gender_estimation.src.factory import get_model


pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(0)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()
            print("____relt____", ret)
            print("____img____", img.shape)

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

model = None
detector = None
img_size = None

def age_prepare():
    global model, detector, img_size

    args = get_args()
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)


def age_detect(face_img, face_coord):
    global model, detector, img_size

    args = get_args()
    # weight_file = args.weight_file
    margin = args.margin
    # image_dir = args.image_dir
    faces = np.empty((1, img_size, img_size, 3))
    faces[0] = cv2.resize(face_img, (img_size, img_size))

    # predict ages and genders of the detected faces
    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    obj = {
        "detected": face_img,
        "predicted_ages": predicted_ages,
        "predicted_genders": predicted_genders,
    }
    return obj 
	
def age_debug(obj, img, coord):
    x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
    # draw results
    new_img = img
    # print("_______detected", obj)
    predicted_ages = obj["predicted_ages"]
    predicted_genders = obj["predicted_genders"]
    detected = [] 
    detected.append(obj["detected"])
    for i, d in enumerate(detected):
        label = "{}, {}".format(int(predicted_ages[i]),
                                "M" if predicted_genders[i][0] < 0.5 else "F")
        print("__debug____", x1, y1, label)
        draw_label(new_img, (x1, y1), label)
    return new_img

if __name__ == '__main__':
    main()
