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

  const gridHelper = new THREE.GridHelper(10, 10)
  scene.add(gridHelper)
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
  const leftEar = landmarks[PoseLandmarkIndexEnum.LEFT_EAR]
  const rightEar = landmarks[PoseLandmarkIndexEnum.RIGHT_EAR]
  const headNode = humanoid.getNormalizedBoneNode("head")
  if (headNode) {
    const earCenter = {
      x: (leftEar.x + rightEar.x) / 2,
      y: (leftEar.y + rightEar.y) / 2,
    }
    const yaw = (nose.x - earCenter.x) * Math.PI * 2
    const pitch = -(nose.y - earCenter.y) * Math.PI * 2
    headNode.rotation.y = yaw
    headNode.rotation.x = pitch
  }

  // 左腕（カメラから見て左 = VRMの右）
  const leftShoulder = landmarks[PoseLandmarkIndexEnum.LEFT_SHOULDER]
  const leftElbow = landmarks[PoseLandmarkIndexEnum.LEFT_ELBOW]
  // 左上腕のZ軸回転（肩の上下動き）
  const leftUpperArmNode = humanoid.getNormalizedBoneNode("leftUpperArm")
  if (leftUpperArmNode) {
    // MediaPipeのY座標は下向きが正なので反転
    // 肩から肘へのベクトルで角度を計算
    const deltaY = -(leftElbow.y - leftShoulder.y) // Y軸を反転
    const deltaX = leftElbow.x - leftShoulder.x
    // Z軸回転（上下の動き）- 水平が0、下が負、上が正
    const zRotation = Math.atan2(deltaY, -deltaX) - Math.PI / 2
    leftUpperArmNode.rotation.z = Math.PI / 2 * +0.8
    // // X軸回転（前後の動き）
    const deltaZ = leftElbow.z - leftShoulder.z
    const horizontalDist = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
    const xRotation = Math.atan2(-deltaZ, horizontalDist)
    // leftUpperArmNode.rotation.x = xRotation
  }
  // 左前腕の曲げ
  const leftLowerArmNode = humanoid.getNormalizedBoneNode("leftLowerArm")
  const leftWrist = landmarks[PoseLandmarkIndexEnum.LEFT_WRIST]
  if (leftLowerArmNode) {
    const shoulder = leftShoulder
    const elbow = leftElbow
    const wrist = leftWrist
    // 上腕ベクトル（肩→肘）
    const upperArm = {
      x: elbow.x - shoulder.x,
      y: elbow.y - shoulder.y,
      z: elbow.z - shoulder.z,
    }
    // 前腕ベクトル（肘→手首）
    const lowerArm = {
      x: wrist.x - elbow.x,
      y: wrist.y - elbow.y,
      z: wrist.z - elbow.z,
    }
    // 内積を使って2つのベクトル間の角度を計算
    const dot = upperArm.x * lowerArm.x + upperArm.y * lowerArm.y + upperArm.z * lowerArm.z
    const upperLength = Math.sqrt(upperArm.x ** 2 + upperArm.y ** 2 + upperArm.z ** 2)
    const lowerLength = Math.sqrt(lowerArm.x ** 2 + lowerArm.y ** 2 + lowerArm.z ** 2)
    // 0で割るのを防ぐ
    if (upperLength === 0 || lowerLength === 0) return 0
    const cosAngle = dot / (upperLength * lowerLength)
    const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)))
    // angleは0（180度、伸ばした状態）からπ（0度、完全に曲げた状態）の範囲
    // VRMの前腕Z軸回転: 0（伸ばした状態）から負の値（曲げた状態）に変換
    // π - angle で反転させて、符号を負にする
    const elbowAngle = -(Math.PI - angle)
    // leftLowerArmNode.rotation.z = elbowAngle
  }
}

// 肘の曲げ角度を計算
// 返り値: 0（伸ばした状態）から負の値（曲げた状態）
const calculateElbowAngle = (
  shoulder: NormalizedLandmark,
  elbow: NormalizedLandmark,
  wrist: NormalizedLandmark
) => {
  // 上腕ベクトル（肩→肘）
  const upperArm = {
    x: elbow.x - shoulder.x,
    y: elbow.y - shoulder.y,
    z: elbow.z - shoulder.z,
  }
  // 前腕ベクトル（肘→手首）
  const lowerArm = {
    x: wrist.x - elbow.x,
    y: wrist.y - elbow.y,
    z: wrist.z - elbow.z,
  }
  // 内積を使って2つのベクトル間の角度を計算
  const dot = upperArm.x * lowerArm.x + upperArm.y * lowerArm.y + upperArm.z * lowerArm.z
  const upperLength = Math.sqrt(upperArm.x ** 2 + upperArm.y ** 2 + upperArm.z ** 2)
  const lowerLength = Math.sqrt(lowerArm.x ** 2 + lowerArm.y ** 2 + lowerArm.z ** 2)
  // 0で割るのを防ぐ
  if (upperLength === 0 || lowerLength === 0) return 0
  const cosAngle = dot / (upperLength * lowerLength)
  const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)))
  // angleは0（180度、伸ばした状態）からπ（0度、完全に曲げた状態）の範囲
  // VRMの前腕Z軸回転: 0（伸ばした状態）から負の値（曲げた状態）に変換
  // π - angle で反転させて、符号を負にする
  return -(Math.PI - angle)
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

  const blendshapeMap: Record<string, string> = {
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
    const vrmExpression = blendshapeMap[shape.categoryName]
    if (vrmExpression && shape.score > 0.1 && vrm.expressionManager) {
      try {
        vrm.expressionManager.setValue(vrmExpression, shape.score)
      } catch (e) {
        // 対応していない表情は無視
      }
    }
  })

  const blinkLeft = blendshapes.find((s) => s.categoryName === "eyeBlinkLeft")?.score || 0
  const blinkRight = blendshapes.find((s) => s.categoryName === "eyeBlinkRight")?.score || 0
  const blink = (blinkLeft + blinkRight) / 2

  if (blink > 0.9 && vrm.expressionManager) {
    try {
      vrm.expressionManager.setValue("blink", 1.0)
    } catch (e) { }
  }

  const smileLeft = blendshapes.find((s) => s.categoryName === "mouthSmileLeft")?.score || 0
  const smileRight = blendshapes.find((s) => s.categoryName === "mouthSmileRight")?.score || 0
  const smile = Math.max(smileLeft, smileRight)

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
