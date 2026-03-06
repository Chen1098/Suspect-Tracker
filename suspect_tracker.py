import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as T
from collections import defaultdict, deque
import warnings
import os

warnings.filterwarnings("ignore")

class EvolvPitchProCPU_Enterprise:
    def __init__(self):
        self.device = 'cpu'
        print(f"[INFO] CPU inference pipeline initialized | Sniper Lock module armed")
        
        self.detector = YOLO('yolo11x.pt')
        self.detector.to(self.device)
        
        # ResNet50 as appearance feature extractor — strip the classification head
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.target_feature_bank = []
        self.suspect_id_in_cam1 = None
        self.current_boxes_for_click = []

    def _reset_tracker(self):
        self.detector = YOLO('yolo11x.pt')
        self.detector.to(self.device)

    def get_feature(self, crop_img):
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.extractor(img_t).numpy().flatten()
        return feat / np.linalg.norm(feat)

    def cosine_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2)

    def crop_with_padding(self, frame, box, pad_ratio=0.08):
        x1, y1, x2, y2 = map(int, box[:4])
        h, w = frame.shape[:2]
        pad_x, pad_y = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        nx2, ny2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return frame[ny1:ny2, nx1:nx2]

    def calc_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    # ================= PHASE 1: Interactive Target Selection =================
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.suspect_id_in_cam1 is None:
            for box, track_id in self.current_boxes_for_click:
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.suspect_id_in_cam1 = track_id
                    print(f"\n[TARGET LOCKED] Track ID: {track_id}")
                    break

    def run_interactive_phase1(self, video1_path):
        self._reset_tracker()
        cap = cv2.VideoCapture(video1_path)
        if not cap.isOpened(): return

        cv2.namedWindow("First Camera (Locate Suspect)", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("First Camera (Locate Suspect)", self.mouse_callback)
        
        while cap.isOpened() and self.suspect_id_in_cam1 is None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            results = self.detector.track(frame, persist=True, classes=[0], imgsz=1280, conf=0.20, verbose=False)
            self.current_boxes_for_click = []
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for b, tid in zip(results[0].boxes.xyxy.numpy(), results[0].boxes.id.numpy()):
                    x1, y1, x2, y2 = map(int, b[:4])
                    self.current_boxes_for_click.append(((x1, y1, x2, y2), tid))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {int(tid)}", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)
                    
            cv2.imshow("First Camera (Locate Suspect)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()
        if self.suspect_id_in_cam1 is not None:
            self.build_feature_bank(video1_path, self.suspect_id_in_cam1)

    # ================= PHASE 1.5: Feature Bank Construction =================
    def build_feature_bank(self, video1_path, suspect_target_id):
        print(f"\n[PHASE 1.5] Building quality-filtered feature bank...")
        self._reset_tracker()
        cap = cv2.VideoCapture(video1_path)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            results = self.detector.track(frame, persist=True, classes=[0], imgsz=1280, conf=0.20, verbose=False)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box in results[0].boxes:
                    if int(box.id[0]) == suspect_target_id and frame_idx % 3 == 0:
                        b_coord = box.xyxy[0].numpy()
                        w, h = b_coord[2] - b_coord[0], b_coord[3] - b_coord[1]
                        # Filter partial detections and non-pedestrian aspect ratios
                        if h > 60 and 0.2 < (w / h) < 0.7:
                            crop = self.crop_with_padding(frame, b_coord)
                            if crop.shape[0] > 40:
                                self.target_feature_bank.append(self.get_feature(crop))
                                self.target_feature_bank.append(self.get_feature(cv2.flip(crop, 1)))
                                print(f"  [+] Feature sample acquired (bank size: {len(self.target_feature_bank)})")
                            
        cap.release()
        print(f"[SUCCESS] Feature bank ready — {len(self.target_feature_bank)} embeddings. Starting cross-camera pursuit.")

    # ================= PHASE 2: Dual-Camera Split-Screen Render =================
    def run_split_screen_render(self, video1_path, video2_path, output_path="evolv_split_demo.mp4"):
        if not self.target_feature_bank: return
            
        print(f"\n[PHASE 2] Dual-stream render: Cam1 (state machine) | Cam2 (sniper lock)...")
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        target_h = 720
        w1, h1 = int(cap1.get(3)), int(cap1.get(4))
        w2, h2 = int(cap2.get(3)), int(cap2.get(4))
        target_w1, target_w2 = int(w1 * (target_h / h1)), int(w2 * (target_h / h2))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w1 + target_w2, target_h))
        
        tracker_cam1 = YOLO('yolo11x.pt')
        tracker_cam2 = YOLO('yolo11x.pt')
        tracker_cam1.to(self.device)
        tracker_cam2.to(self.device)

        # Cam1: sliding-window similarity state machine keyed by tracker ID
        track_states_cam1 = defaultdict(lambda: {"sims": deque(maxlen=5), "state": "UNKNOWN", "miss": 0})
        
        # Cam2: ID-agnostic sniper lock — persists purely on appearance + spatial continuity
        cam2_locked_box = None
        cam2_missed_frames = 0
        cam2_current_sim = 0.0

        curr, max_frames = 0, max(int(cap1.get(7)), int(cap2.get(7)))

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 and not ret2: break
            curr += 1
            if curr % 5 == 0: print(f"  -> Rendering: {curr}/{max_frames}")

            # ======= CAM 1 =======
            if ret1:
                res1 = tracker_cam1.track(frame1, persist=True, classes=[0], imgsz=1280, conf=0.20, tracker="botsort.yaml", verbose=False)
                if res1[0].boxes is not None and res1[0].boxes.id is not None:
                    active_ids = set()
                    boxes = res1[0].boxes.xyxy.numpy()
                    tids = res1[0].boxes.id.numpy()
                    
                    for box, tid in zip(boxes, tids):
                        tid = int(tid)
                        active_ids.add(tid)
                        crop = self.crop_with_padding(frame1, box)
                        if crop.shape[0] < 40: continue
                        
                        feat = self.get_feature(crop)
                        max_sim = max([self.cosine_similarity(b, feat) for b in self.target_feature_bank])
                        
                        track_states_cam1[tid]["sims"].append(max_sim)
                        track_states_cam1[tid]["miss"] = 0
                        
                        history = list(track_states_cam1[tid]["sims"])
                        mean_sim = sum(history) / len(history)
                        
                        if track_states_cam1[tid]["state"] == "UNKNOWN" and len(history) >= 3 and mean_sim > 0.82:
                            track_states_cam1[tid]["state"] = "SUSPECT"
                        elif track_states_cam1[tid]["state"] == "SUSPECT" and len(history) == 5 and mean_sim < 0.70:
                            track_states_cam1[tid]["state"] = "UNKNOWN"

                        x1, y1, x2, y2 = map(int, box[:4])
                        if track_states_cam1[tid]["state"] == "SUSPECT":
                            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.rectangle(frame1, (x1, y1-35), (x1+250, y1), (0, 0, 255), -1)
                            cv2.putText(frame1, "SOURCE TARGET", (x1+5, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                        else:
                            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    for tid in list(track_states_cam1.keys()):
                        if tid not in active_ids:
                            track_states_cam1[tid]["miss"] += 1
                            if track_states_cam1[tid]["miss"] > 30: del track_states_cam1[tid]
                frame1_rs = cv2.resize(frame1, (target_w1, target_h))
            else:
                frame1_rs = np.zeros((target_h, target_w1, 3), dtype=np.uint8)

            # ======= CAM 2 =======
            if ret2:
                # Higher confidence threshold to suppress ghost detections; no tracker ID dependency
                res2 = tracker_cam2.predict(frame2, classes=[0], imgsz=1280, conf=0.35, verbose=False)
                best_box = None
                best_score = -1.0
                best_sim = 0.0

                if res2[0].boxes is not None and len(res2[0].boxes) > 0:
                    boxes = res2[0].boxes.xyxy.numpy()
                    
                    for box in boxes:
                        crop = self.crop_with_padding(frame2, box)
                        if crop.shape[0] < 40: continue
                        
                        feat = self.get_feature(crop)
                        sim = max([self.cosine_similarity(b, feat) for b in self.target_feature_bank])
                        
                        if cam2_locked_box is None:
                            # Acquisition mode: rank purely by ReID similarity
                            if sim > best_score:
                                best_score = sim
                                best_box = box
                                best_sim = sim
                        else:
                            # Tracking mode: joint scoring — IoU anchors spatial continuity,
                            # ReID handles re-identification through occlusion
                            iou = self.calc_iou(cam2_locked_box, box)
                            score = sim + (iou * 1.5)
                            
                            if score > best_score:
                                best_score = score
                                best_box = box
                                best_sim = sim

                if cam2_locked_box is None:
                    if best_box is not None and best_sim > 0.76:
                        cam2_locked_box = best_box
                        cam2_current_sim = best_sim
                        cam2_missed_frames = 0
                        print(f"  [ACTION] Sniper lock acquired (Sim: {best_sim:.2f})")
                else:
                    if best_box is not None and (best_score > 0.55):
                        cam2_locked_box = best_box
                        cam2_current_sim = best_sim
                        cam2_missed_frames = 0
                    else:
                        cam2_missed_frames += 1
                        if cam2_missed_frames > 20:
                            cam2_locked_box = None
                            print(f"  [INFO] Lock lost — re-entering acquisition mode...")

                if res2[0].boxes is not None:
                    for box in res2[0].boxes.xyxy.numpy():
                        if cam2_locked_box is not None and np.array_equal(box, best_box):
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.rectangle(frame2, (x1, y1-35), (x1+250, y1), (0, 0, 255), -1)
                            cv2.putText(frame2, f"SNIPER LOCK | {cam2_current_sim:.2f}", (x1+5, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                        else:
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 1)

                frame2_rs = cv2.resize(frame2, (target_w2, target_h))
            else:
                frame2_rs = np.zeros((target_h, target_w2, 3), dtype=np.uint8)

            # ======= Composite Output =======
            combined = np.hstack((frame1_rs, frame2_rs))
            cv2.line(combined, (target_w1, 0), (target_w1, target_h), (255, 255, 255), 3)
            cv2.rectangle(combined, (0, 0), (target_w1 + target_w2, 40), (0, 0, 0), -1)
            cv2.putText(combined, "First Camera (Locate Suspect)", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combined, "Second Camera (Lock On Suspect)", (target_w1 + 10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            out.write(combined)
            
        cap1.release()
        cap2.release()
        out.release()
        print(f"\n[DONE] Demo rendered: {output_path}")

if __name__ == "__main__":
    system = EvolvPitchProCPU_Enterprise()
    v1, v2 = "video1.mp4", "video2.mp4"
    if os.path.exists(v1) and os.path.exists(v2):
        system.run_interactive_phase1(v1)
        system.run_split_screen_render(v1, v2)
    else:
        print("[ERROR] Video files not found.")