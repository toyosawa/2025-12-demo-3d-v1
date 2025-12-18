import * as THREE from "three"
import {
  FilesetResolver,
  PoseLandmarker,
  FaceLandmarker,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision"
import { VRMLoaderPlugin, VRM } from "@pixiv/three-vrm"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader"

// Three.jsのセットアップ
export const setupThree = (canvas: HTMLCanvasElement) => {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x212121)

  const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100)
  camera.position.set(0, 1.4, 3)

  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(1, 1, 1)
  scene.add(light)
  scene.add(new THREE.AmbientLight(0xffffff, 0.5))

  const grid_helper = new THREE.GridHelper(10, 10)
  scene.add(grid_helper)
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(window.devicePixelRatio)

  return { scene, camera, renderer }
}

// VRMモデルのロード
export const setupVRMFromURL = async (url: string) => {
  const loader = new GLTFLoader()
  loader.register((parser) => new VRMLoaderPlugin(parser))
  const gltf = await loader.loadAsync(url)
  const vrm = gltf.userData.vrm as VRM
  return vrm
}

// MediaPipeのセットアップ
export const setupMediaPipe = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm",
  )

  const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
    },
    runningMode: "VIDEO",
    numPoses: 1,
  })

  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO",
    numFaces: 1,
  })
  return { poseLandmarker, faceLandmarker }
}

// ポーズをVRMに適用
export const applyPoseToVRM = (
  results: { landmarks: NormalizedLandmark[][] },
  vrm: VRM
) => {
  if (!results.landmarks || results.landmarks.length === 0) return
  if (!vrm) return

  const landmarks = results.landmarks[0]
  const humanoid = vrm.humanoid

  // 頭部
  const nose = landmarks[PoseLandmarkIndexEnum.NOSE]
  const left_ear = landmarks[PoseLandmarkIndexEnum.LEFT_EAR]
  const right_ear = landmarks[PoseLandmarkIndexEnum.RIGHT_EAR]
  const head_node = humanoid.getNormalizedBoneNode("head")
  if (head_node) {
    const ear_center = {
      x: (left_ear.x + right_ear.x) / 2,
      y: (left_ear.y + right_ear.y) / 2,
    }
    head_node.rotation.y = (nose.x - ear_center.x) * Math.PI * 2
    head_node.rotation.x = -(nose.y - ear_center.y) * Math.PI * 2
  }

  // 左腕（カメラから見て左 = VRMの右）
  const left_shoulder = landmarks[PoseLandmarkIndexEnum.LEFT_SHOULDER]
  const left_elbow = landmarks[PoseLandmarkIndexEnum.LEFT_ELBOW]
  // 左上腕のZ軸回転（肩の上下動き）
  const left_upper_arm_node = humanoid.getNormalizedBoneNode("leftUpperArm")
  if (left_upper_arm_node) {
    // MediaPipeのY座標は下向きが正なので反転
    // 肩から肘へのベクトルで角度を計算
    const delta_y = -(left_elbow.y - left_shoulder.y) // Y軸を反転
    const delta_x = left_elbow.x - left_shoulder.x
    // Z軸回転（上下の動き）- 水平が0、下が負、上が正
    const z_rotation = Math.atan2(delta_y, -delta_x) - Math.PI / 2
    left_upper_arm_node.rotation.z = Math.PI / 2 * +0.8
    // // X軸回転（前後の動き）
    const delta_z = left_elbow.z - left_shoulder.z
    const horizontal_dist = Math.sqrt(delta_x * delta_x + delta_y * delta_y)
    const x_rotation = Math.atan2(-delta_z, horizontal_dist)
    // left_upper_arm_node.rotation.x = x_rotation
  }
  // 左前腕の曲げ
  const left_lower_arm_node = humanoid.getNormalizedBoneNode("leftLowerArm")
  const left_wrist = landmarks[PoseLandmarkIndexEnum.LEFT_WRIST]
  if (left_lower_arm_node) {
    const shoulder = left_shoulder
    const elbow = left_elbow
    const wrist = left_wrist
    // 上腕ベクトル（肩→肘）
    const upper_arm = {
      x: elbow.x - shoulder.x,
      y: elbow.y - shoulder.y,
      z: elbow.z - shoulder.z,
    }
    // 前腕ベクトル（肘→手首）
    const lower_arm = {
      x: wrist.x - elbow.x,
      y: wrist.y - elbow.y,
      z: wrist.z - elbow.z,
    }
    // 内積を使って2つのベクトル間の角度を計算
    const dot = upper_arm.x * lower_arm.x + upper_arm.y * lower_arm.y + upper_arm.z * lower_arm.z
    const upper_length = Math.sqrt(upper_arm.x ** 2 + upper_arm.y ** 2 + upper_arm.z ** 2)
    const lower_length = Math.sqrt(lower_arm.x ** 2 + lower_arm.y ** 2 + lower_arm.z ** 2)
    // 0で割るのを防ぐ
    if (upper_length === 0 || lower_length === 0) return 0
    const cos_angle = dot / (upper_length * lower_length)
    const angle = Math.acos(Math.max(-1, Math.min(1, cos_angle)))
    // angleは0（180度、伸ばした状態）からπ（0度、完全に曲げた状態）の範囲
    // VRMの前腕Z軸回転: 0（伸ばした状態）から負の値（曲げた状態）に変換
    // π - angle で反転させて、符号を負にする
    const elbow_angle = -(Math.PI - angle)
    // left_lower_arm_node.rotation.z = elbow_angle
  }
}

// 表情をVRMに適用
export const applyFaceToVRM = (results: {
  faceBlendshapes: {
    categories: {
      categoryName: string
      score: number
    }[]
  }[]
}, vrm: VRM) => {
  if (!results.faceBlendshapes || results.faceBlendshapes.length === 0) return
  if (!vrm.expressionManager) return
  const blendshapes = results.faceBlendshapes[0].categories

  const blendshape_map: Record<string, string> = {
    eyeBlinkLeft: "blinkLeft",
    eyeBlinkRight: "blinkRight",
    jawOpen: "aa",
    mouthSmileLeft: "joy",
    mouthSmileRight: "joy",
    browDownLeft: "angry",
    browDownRight: "angry",
    mouthFrownLeft: "sorrow",
    mouthFrownRight: "sorrow",
  }

  blendshapes.forEach((shape) => {
    const vrm_expression = blendshape_map[shape.categoryName]
    if (vrm_expression && shape.score > 0.1 && vrm.expressionManager) {
      try {
        vrm.expressionManager.setValue(vrm_expression, shape.score)
      } catch (e) {
        // 対応していない表情は無視
      }
    }
  })

  const blink_left = blendshapes.find((s) => s.categoryName === "eyeBlinkLeft")?.score || 0
  const blink_right = blendshapes.find((s) => s.categoryName === "eyeBlinkRight")?.score || 0
  const blink = (blink_left + blink_right) / 2

  if (blink > 0.9 && vrm.expressionManager) {
    try {
      vrm.expressionManager.setValue("blink", 1.0)
    } catch (e) { }
  }

  const smile_left = blendshapes.find((s) => s.categoryName === "mouthSmileLeft")?.score || 0
  const smile_right = blendshapes.find((s) => s.categoryName === "mouthSmileRight")?.score || 0
  const smile = Math.max(smile_left, smile_right)

  if (smile > 0.3 && vrm.expressionManager) {
    try {
      vrm.expressionManager.setValue("happy", smile)
    } catch (e) { }
  }
}

// MediaPipe Pose Landmarkのインデックス定義
const PoseLandmarkIndexEnum = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const
