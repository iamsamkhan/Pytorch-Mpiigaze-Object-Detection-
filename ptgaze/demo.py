import datetime
import logging
import pathlib
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from .common import Face, FacePartsName, Visualizer
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')} # Press Esc or q to terminate loop

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()

        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        self.current_datetime = datetime.datetime.now() # date and time at start of running file
        self.store_data_dir = self.config.demo.data_output_dir
        #''
        self.store_path = self._generate_path()


    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError
        
    def _generate_path(self) -> str:
        date = "-".join([ str(self.current_datetime.day).zfill(2), str(self.current_datetime.month).zfill(2), str(self.current_datetime.year) ])
        time = "-".join([ str(self.current_datetime.hour).zfill(2), str(self.current_datetime.minute).zfill(2), str(self.current_datetime.second).zfill(2) ])
        video_name = "_".join([pathlib.Path(self.config.demo.video_path).stem, date, time]) + ".csv"
        store_path = "/".join([self.store_data_dir, video_name])
        return store_path
    
    @staticmethod
    def _get_frame_point(pt0 , pt1, width, height) -> tuple:
        num = pt1[1]-pt0[1]
        deno = pt1[0]-pt0[0]

        if (round(num,0) == 0) or (round(deno,0)) == 0:
            return pt1

        slope = num/deno
        pt = [0,0]

        if num <= 0 and deno >= 0:
            pt[1] = 0;
            if (pt0[0]-(pt0[1]/slope)) > width:
                pt[0] = width
                pt[1] = pt0[1] + slope*(pt[0] - pt0[0])
            else:
                pt[0] = pt0[0] - pt0[1]/slope
        elif num < 0 and deno < 0:
            pt[1] = 0;
            if (pt0[0]-(pt0[1]/slope)) < 0:
                pt[0] = 0
                pt[1] = pt0[1] - slope*pt0[0]
            else:
                pt[0] = pt0[0] - pt0[1]/slope
        elif num > 0 and deno < 0:
            pt[0] = 0;
            if (pt0[1]-(slope*pt0[0])) > height:
                pt[1] = height
                pt[0] = pt0[0] + (pt[1] - pt0[1])/slope
            else:
                pt[1] = pt0[1] - slope*pt0[0]
        else:   #fourth quadrant eqv
            pt[0] = width
            if (pt0[1] + slope*(width - pt0[0])) > height:
                pt[1] = height
                pt[0] = pt0[0] + (pt[1] - pt0[1])/slope
            else:
                pt[1] = pt0[1] + slope*(pt[0] - pt0[0])
        
        pt[0] = int(round(pt[0],0))
        pt[1] = int(round(pt[1],0))

        return (pt[0],pt[1])

    def _run_on_image(self):
        
        # Create a DataFrame to store gaze data
        cols = ['face','yaw','pitch','classification']
        df = pd.DataFrame(columns=cols)
        
        # Raise exception if image has no data
        try:
            image = cv2.imread(self.config.demo.image_path)
        except:
            df.to_csv(self.store_path)
            raise ValueError("image not found.")
        
        # Raise exception if image has no faces
        face_ls = self._process_image(image)
        if (face_ls==[]):
            df.to_csv(self.store_path)
            raise ValueError("No faces detected.")

        while True:
            key_pressed = self._wait_key()
            if self.stop:
                break
            if key_pressed:
                # Process the frame and obtain a list of detected faces
                face_ls = self._process_image(image)
            

        y = 100 # y coordinate to position direction text in image
        num = 1 # face no.
        for ls in face_ls:
            # Project the face points onto the frame
            points2d = self.visualizer._camera.project_points(np.vstack([ls[2],ls[3]]))
            points2d = self.visualizer._convert_pt(points2d)
            pt0 = points2d[0]
            pt1 = points2d[1]
            pt = self._get_frame_point(pt0 , pt1, self.gaze_estimator.camera.width, self.gaze_estimator.camera.height)
            
            # Draw circles at the projected gaze vector points
            cv2.circle(self.visualizer.image, pt0, 5, (255,0,0), -1)
            cv2.circle(self.visualizer.image, pt1, 5, (0,0,255), -1)
            cv2.circle(self.visualizer.image, pt, 10, (0,255,0), -1)

            clf = None
            if (abs(pt0[0]-pt1[0])<=7):
                clf = "FRONT"
            else:
                right = -1
                up = -1
                
                # Determine the direction classification based on the position of the projected point
                if (pt[1] < self.gaze_estimator.camera.height/6):
                    up = 1
                elif (pt[1] > self.gaze_estimator.camera.height*(5/6)):
                    up = 0

                if (pt[0] < self.gaze_estimator.camera.width/6):
                    right = 1
                    
                elif (pt[0] > self.gaze_estimator.camera.width*(5/6)):
                    right = 0
                    
                if up == 1:
                    if right == -1:
                        clf = "UP"
                    elif right == 1:
                        clf = "TOP_RIGHT"
                    elif right == 0:
                        clf = "TOP_LEFT"
                elif up == 0:
                    if right == -1:
                        clf = "DOWN"
                    elif right == 1:
                        clf = "BOTTOM_RIGHT"
                    elif right == 0:
                        clf = "BOTTOM_LEFT"
                else:
                    if right == 1:
                        clf = "RIGHT"
                    elif right == 0:
                        clf = "LEFT"
                
            # Add the gaze data to the DataFrame    
            add = pd.DataFrame({'face': [num], 'yaw': [round(ls[0],2)], 'pitch': [round(ls[1],2)], 'classification': [clf]})
            df = pd.concat([df,add], ignore_index=True)
            num += 1
            y += 50 

            # Display frame if needed
            if self.config.demo.display_on_screen:
                
                if(clf=="FRONT"):
                    cv2.putText(self.visualizer.image, f"face{num}:FRONT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "UP"):
                    cv2.putText(self.visualizer.image, f"face{num}:UP", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "TOP_RIGHT"):
                    cv2.putText(self.visualizer.image, f"face{num}:TOP_RIGHT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "TOP_LEFT"):
                    cv2.putText(self.visualizer.image, f"face{num}:TOP_LEFT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "DOWN"):
                    cv2.putText(self.visualizer.image, f"face{num}:DOWN", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "BOTTOM_RIGHT"):
                    cv2.putText(self.visualizer.image, f"face{num}:BOTTOM_RIGHT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "BOTTOM_LEFT"):
                    cv2.putText(self.visualizer.image, f"face{num}:BOTTOM_LEFT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf == "RIGHT"):
                    cv2.putText(self.visualizer.image, f"face{num}:RIGHT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif(clf=="LEFT"):
                    cv2.putText(self.visualizer.image, f"face{num}:LEFT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('image', self.visualizer.image)

        # Save data to video output directory
        df.to_csv(self.store_path)

        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        # Create a DataFrame to store gaze data
        cols = ['face', 'time_stamp', 'yaw', 'pitch', 'classification']
        df = pd.DataFrame(columns=cols)

        # Get the frames per second (fps) of the video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"fps: {fps}")

        frame_counter = 0
        flag = 0

        while self.cap.isOpened():
            # Check if the display on screen is enabled and wait for a key press
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            # Read the next frame from the video
            ok, frame = self.cap.read()

            # Check if the frame was read successfully
            if not ok:
                if frame_counter == 0:
                    df.to_csv(self.store_path, index=False, mode='w')
                    raise ValueError("Empty Video")
                break

            frame_counter += 1

            # Process the frame and obtain a list of detected faces
            face_ls = self._process_image(frame)

            # Check if no faces were detected in the frame
            if face_ls != []:
                flag = 1
            else:
                if self.config.demo.display_on_screen:
                    cv2.imshow('frame', self.visualizer.image)
                continue

            y = 50  # y coordinate for direction text if frame is displayed
            num = 1  # face number

            for ls in face_ls:
                # Project the face points onto the frame
                points2d = self.visualizer._camera.project_points(np.vstack([ls[2], ls[3]]))
                points2d = self.visualizer._convert_pt(points2d)
                pt0 = points2d[0]
                pt1 = points2d[1]
                pt = self._get_frame_point(pt0, pt1, self.gaze_estimator.camera.width, self.gaze_estimator.camera.height)

                # Draw circles at the projected gaze vector points
                cv2.circle(self.visualizer.image, pt0, 5, (255, 0, 0), -1)
                cv2.circle(self.visualizer.image, pt1, 5, (0, 0, 255), -1)
                cv2.circle(self.visualizer.image, pt, 10, (0, 255, 0), -1)

                clf = None

                if abs(pt0[0] - pt1[0]) <= 7:
                    clf = "FRONT"
                else:
                    right = -1
                    up = -1

                    # Determine the direction classification based on the position of the projected point
                    if pt[1] < self.gaze_estimator.camera.height / 6:
                        up = 1
                    elif pt[1] > self.gaze_estimator.camera.height * (5 / 6):
                        up = 0

                    if pt[0] < self.gaze_estimator.camera.width / 6:
                        right = 1
                    elif pt[0] > self.gaze_estimator.camera.width * (5 / 6):
                        right = 0

                    if up == 1:
                        if right == -1:
                            clf = "UP"
                        elif right == 1:
                            clf = "TOP_RIGHT"
                        elif right == 0:
                            clf = "TOP_LEFT"
                    elif up == 0:
                        if right == -1:
                            clf = "DOWN"
                        elif right == 1:
                            clf = "BOTTOM_RIGHT"
                        elif right == 0:
                            clf = "BOTTOM_LEFT"
                    else:
                        if right == 1:
                            clf = "RIGHT"
                        elif right == 0:
                            clf = "LEFT"

                # Add the gaze data to the DataFrame
                add = pd.DataFrame({'face': [num], 'time_stamp': [round(frame_counter * (1 / fps), 2)], 'yaw': [round(ls[0], 2)], 'pitch': [round(ls[1], 2)], 'classification': [clf]})
                df = pd.concat([df, add], ignore_index=True)

                # Display frame if needed
                if self.config.demo.display_on_screen:
                    
                    if(clf=="FRONT"):
                        cv2.putText(self.visualizer.image, f"face{num}:FRONT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "UP"):
                        cv2.putText(self.visualizer.image, f"face{num}:UP", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "TOP_RIGHT"):
                        cv2.putText(self.visualizer.image, f"face{num}:TOP_RIGHT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "TOP_LEFT"):
                        cv2.putText(self.visualizer.image, f"face{num}:TOP_LEFT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "DOWN"):
                        cv2.putText(self.visualizer.image, f"face{num}:DOWN", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "BOTTOM_RIGHT"):
                        cv2.putText(self.visualizer.image, f"face{num}:BOTTOM_RIGHT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "BOTTOM_LEFT"):
                        cv2.putText(self.visualizer.image, f"face{num}:BOTTOM_LEFT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf == "RIGHT"):
                        cv2.putText(self.visualizer.image, f"face{num}:RIGHT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif(clf=="LEFT"):
                        cv2.putText(self.visualizer.image, f"face{num}:LEFT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow('frame', self.visualizer.image)
                    
                num += 1 #for new face
                y += 50 #shift display text for new face

        # Save data to video output directory
        df.to_csv(self.store_path, index=False, mode='w')
            
        # Raise an exception if no faces were detected
        if flag == 0:
            raise ValueError("No faces detected.")

        self.cap.release()
        if self.config.demo.display_on_screen:
            cv2.destroyAllWindows()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> list:
        image = cv2.resize(image,(self.gaze_estimator.camera.width,self.gaze_estimator.camera.height))
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)
        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        face_ls = []
        for face in faces:
            # Append face to face_ls
            face_ls.append(self._draw_gaze_vector(face))
            
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            #self.visualizer.image = self.visualizer.image[:, ::-1]
            self.visualizer.image = cv2.cvtColor(self.visualizer.image, cv2.COLOR_BGR2RGB)
        if self.writer:
            self.writer.write(self.visualizer.image)
        
        # Return list of faces
        return face_ls


    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        # Resize frame width
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        ht = cv2.CAP_PROP_FRAME_HEIGHT
        wd = cv2.CAP_PROP_FRAME_WIDTH
        self.gaze_estimator.camera.width = 640
        self.gaze_estimator.camera.height = int((wd/ht)*self.gaze_estimator.camera.width)
        print(f"set camera width: {self.gaze_estimator.camera.width} and camera height: {self.gaze_estimator.camera.height}")
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> list:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                #logger.info(
                    #f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            #logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')

            # Return list of yaw, pitch and endpoints of gaze vector
            return [yaw, pitch, face.center, face.center + length * face.gaze_vector]
        else:
            raise ValueError
