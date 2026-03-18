import cv2
import math
from pupil_apriltags import Detector


class AprilTagNavigator:
    """
    This class handles AprilTag detection for the robot's basket docking task.

    Main responsibilities:
    1. Detect AprilTags in a frame
    2. Keep only the two basket tag IDs we care about
    3. Choose the closest valid tag
    4. Compute steering info so the robot can move toward it
    5. Decide when the robot is 'locked' onto the tag
    """

    def __init__(
        self,
        target_tag_ids,
        camera_params,
        tag_size_m,
        frame_size=(640, 480),
        lock_distance_cm=15.0,
        lock_center_tolerance_px=40,
        lock_yaw_deg=10.0,
        tag_families="tagStandard41h12"
    ):
        """
        Parameters:
        ----------
        target_tag_ids : set or list
            The tag IDs that correspond to the basket.
            Example: {1, 2}

        camera_params : tuple
            Camera intrinsic parameters in the form:
            (fx, fy, cx, cy)

        tag_size_m : float
            Real size of the printed tag in metres.
            Example: 0.08 for an 8 cm tag.

        frame_size : tuple
            Size of the camera frame as (width, height)

        lock_distance_cm : float
            If the robot is within this distance, it may be considered close enough to lock.

        lock_center_tolerance_px : float
            How close the tag centre must be to the image centre in pixels.

        lock_yaw_deg : float
            How 'head-on' the robot must be facing the tag.

        tag_families : str
            The AprilTag family being used.
        """

        # Store input settings for later use
        self.target_tag_ids = set(target_tag_ids)
        self.camera_params = camera_params
        self.tag_size_m = tag_size_m
        self.frame_width, self.frame_height = frame_size

        # Conditions used to decide if the robot is aligned enough to "lock"
        self.lock_distance_cm = lock_distance_cm
        self.lock_center_tolerance_px = lock_center_tolerance_px
        self.lock_yaw_deg = lock_yaw_deg

        # Create the AprilTag detector object
        # This comes from the pupil_apriltags library
        self.detector = Detector(
            families=tag_families,
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

    def _compute_yaw_deg(self, rotation_matrix):
        """
        Estimate the yaw angle of the detected tag from its rotation matrix.

        Yaw here gives an idea of how rotated the tag is relative to the camera.
        Ideally, when docking perpendicularly, yaw should be close to 0.

        Returns:
        --------
        float : yaw angle in degrees
        """

        # This is a practical approximation for yaw
        yaw_rad = math.atan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
        return math.degrees(yaw_rad)

    def _normalise_magnitude(self, distance_cm, max_distance_cm=100.0):
        """
        Convert distance into a movement magnitude between 0.2 and 1.0.

        This is useful because your main motor code likely expects a movement
        "strength" rather than raw centimetres.

        If very far away, magnitude approaches 1.0
        If closer, magnitude becomes smaller but not below 0.2
        """

        mag = min(distance_cm / max_distance_cm, 1.0)
        return max(0.2, mag)

    def _is_locked(self, chosen_tag):
        """
        Decide whether the robot has achieved a good docking alignment.

        The robot is considered locked if:
        1. It is close enough
        2. The tag is near the centre of the image
        3. The tag is roughly facing the robot head-on
        """

        # Find how far the tag is from the middle of the camera image
        frame_center_x = self.frame_width / 2
        centre_error_px = chosen_tag["centre_x"] - frame_center_x

        return (
            chosen_tag["distance_cm"] <= self.lock_distance_cm
            and abs(centre_error_px) <= self.lock_center_tolerance_px
            and abs(chosen_tag["yaw_deg"]) <= self.lock_yaw_deg
        )

    def process_frame(self, frame):
        """
        Process one camera frame and return docking/navigation info.

        Output format:
        --------------
        {
            "found": True/False,
            "lockedTag": True/False,
            "moveTargetTag": {"angle": ..., "magnitude": ...} or None,
            "chosen_tag": {...} or None,
            "all_tags": [...]
        }

        Notes:
        ------
        - If no valid basket tags are found, "found" is False
        - If a tag is found but not locked, moveTargetTag is returned
        - If a tag is locked, moveTargetTag becomes None because the robot
          should switch to the final docking sequence
        """

        # Convert the colour image to grayscale because AprilTag detection
        # works on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all AprilTags in the frame and estimate their pose
        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size_m
        )

        # We will store only the valid basket tags here
        valid_tags = []

        # Loop through every detected tag
        for tag in results:
            # Ignore tags that are not one of our basket tags
            if tag.tag_id not in self.target_tag_ids:
                continue

            # pose_t is the translation vector from camera to tag
            # It is given in metres
            tx = float(tag.pose_t[0][0])   # left/right offset
            tz = float(tag.pose_t[2][0])   # forward distance

            # Convert metres to centimetres for easier understanding
            x_cm = tx * 100.0
            z_cm = tz * 100.0

            # Straight-line distance from camera to tag
            distance_cm = math.sqrt(x_cm**2 + z_cm**2)

            # Estimate tag yaw angle
            yaw_deg = self._compute_yaw_deg(tag.pose_R)

            # Pixel centre of the tag in the image
            centre_x = int(tag.center[0])
            centre_y = int(tag.center[1])

            # Save all useful info about this tag
            valid_tags.append({
                "tag_id": int(tag.tag_id),
                "centre_x": centre_x,
                "centre_y": centre_y,
                "x_cm": x_cm,
                "z_cm": z_cm,
                "distance_cm": distance_cm,
                "yaw_deg": yaw_deg
            })

        # If we found no valid basket tags, return an empty result
        if not valid_tags:
            return {
                "found": False,
                "lockedTag": False,
                "moveTargetTag": None,
                "chosen_tag": None,
                "all_tags": []
            }

        # Pick the closest valid tag
        chosen_tag = min(valid_tags, key=lambda t: t["distance_cm"])

        # Compute steering angle toward the chosen tag
        # atan2(x, z) gives the angle needed to turn toward the tag
        angle_deg = math.degrees(math.atan2(chosen_tag["x_cm"], chosen_tag["z_cm"]))

        # Convert distance into a movement magnitude for the motor controller
        magnitude = self._normalise_magnitude(chosen_tag["distance_cm"])

        # Decide if the robot is aligned and close enough to "lock"
        locked = self._is_locked(chosen_tag)

        # If locked, we don't need a normal movement target anymore
        # because the main code should switch to the final approach logic
        if locked:
            move_target = None
        else:
            move_target = {
                "angle": angle_deg,
                "magnitude": magnitude
            }

        return {
            "found": True,
            "lockedTag": locked,
            "moveTargetTag": move_target,
            "chosen_tag": chosen_tag,
            "all_tags": valid_tags
        }

    def draw_debug(self, frame, result):
        """
        Draw visual debugging information onto the frame.

        This is useful for testing:
        - shows all detected valid tags
        - highlights the chosen target tag
        - displays lockedTag status
        """

        output = frame.copy()

        # Draw all valid tags
        for tag in result["all_tags"]:
            cx, cy = tag["centre_x"], tag["centre_y"]

            # Red dot at tag centre
            cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

            # Text showing tag ID and distance
            text = f"ID {tag['tag_id']} d={tag['distance_cm']:.1f}cm"
            cv2.putText(
                output,
                text,
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Highlight the chosen tag more clearly
        if result["chosen_tag"] is not None:
            cx = result["chosen_tag"]["centre_x"]
            cy = result["chosen_tag"]["centre_y"]

            # Blue if still approaching, yellow if locked
            colour = (0, 255, 255) if result["lockedTag"] else (255, 0, 0)
            cv2.circle(output, (cx, cy), 20, colour, 3)

        # Show lock status in the corner
        status = f"lockedTag={result['lockedTag']}"
        cv2.putText(
            output,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return output