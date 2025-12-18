import * as THREE from "three"
import { FilesetResolver, PoseLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision"
import { VRMLoaderPlugin, VRM } from "@pixiv/three-vrm"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader"

// 型定義
interface Landmark {
  x: number
  y: number
  z: number
  visibility?: number
}

// MediaPipe Pose Landmarkのインデックス定義
const PoseLandmark = {
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

interface PoseResults {
  landmarks: Landmark[][]
}

interface BlendshapeCategory {
  categoryName: string
  score: number
}

interface FaceResults {
  faceBlendshapes: {
    categories: BlendshapeCategory[]
  }[]
}

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
export const applyPoseToVRM = (results: PoseResults, vrm: VRM) => {
  if (!results.landmarks || results.landmarks.length === 0) return
  if (!vrm) return

  const landmarks = results.landmarks[0]
  const humanoid = vrm.humanoid

  // 左腕
  const leftShoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
  const leftElbow = landmarks[PoseLandmark.LEFT_ELBOW]
  const leftWrist = landmarks[PoseLandmark.LEFT_WRIST]
  const leftUpperArmNode = humanoid.getNormalizedBoneNode("leftUpperArm")
  if (leftUpperArmNode) {
    // 上腕の回転
    const armAngles = calculateArmAngles(leftShoulder, leftElbow, leftWrist)
    leftUpperArmNode.rotation.z = armAngles.z
    leftUpperArmNode.rotation.x = -armAngles.x
  }
  const leftLowerArmNode = humanoid.getNormalizedBoneNode("leftLowerArm")
  if (leftLowerArmNode) {
    // 前腕の回転
    const elbowAngle = calculateElbowAngle(leftShoulder, leftElbow, leftWrist)
    leftLowerArmNode.rotation.z = elbowAngle
  }

  // 右腕
  const rightShoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
  const rightElbow = landmarks[PoseLandmark.RIGHT_ELBOW]
  const rightWrist = landmarks[PoseLandmark.RIGHT_WRIST]
  const rightUpperArmNode = humanoid.getNormalizedBoneNode("rightUpperArm")
  if (rightUpperArmNode) {
    // 上腕の回転
    const armAngles = calculateArmAngles(rightShoulder, rightElbow, rightWrist)
    rightUpperArmNode.rotation.z = -armAngles.z
    rightUpperArmNode.rotation.x = armAngles.x
  }
  const rightLowerArmNode = humanoid.getNormalizedBoneNode("rightLowerArm")
  if (rightLowerArmNode) {
    // 前腕の回転
    const elbowAngle = calculateElbowAngle(rightShoulder, rightElbow, rightWrist)
    rightLowerArmNode.rotation.z = -elbowAngle
  }

  // 頭部
  const nose = landmarks[PoseLandmark.NOSE]
  const leftEar = landmarks[PoseLandmark.LEFT_EAR]
  const rightEar = landmarks[PoseLandmark.RIGHT_EAR]
  const headNode = humanoid.getNormalizedBoneNode("head")
  if (headNode) {
    // 頭の回転
    const headRotation = calculateHeadRotation(nose, leftEar, rightEar)
    headNode.rotation.y = headRotation.y
    headNode.rotation.x = headRotation.x
  }
}

// 腕の角度計算
const calculateArmAngles = (shoulder: Landmark, elbow: Landmark, wrist: Landmark) => {
  const dy = elbow.y - shoulder.y
  const dx = elbow.x - shoulder.x
  const zRotation = Math.atan2(dy, dx)

  const dz = elbow.z - shoulder.z
  const horizontalDist = Math.sqrt(dx * dx + dy * dy)
  const xRotation = Math.atan2(dz, horizontalDist)

  return { z: zRotation, x: xRotation }
}

// 肘の曲げ角度を計算
const calculateElbowAngle = (shoulder: Landmark, elbow: Landmark, wrist: Landmark) => {
  const upperArm = {
    x: elbow.x - shoulder.x,
    y: elbow.y - shoulder.y,
    z: elbow.z - shoulder.z,
  }

  const lowerArm = {
    x: wrist.x - elbow.x,
    y: wrist.y - elbow.y,
    z: wrist.z - elbow.z,
  }

  const dot = upperArm.x * lowerArm.x + upperArm.y * lowerArm.y + upperArm.z * lowerArm.z
  const upperLength = Math.sqrt(upperArm.x ** 2 + upperArm.y ** 2 + upperArm.z ** 2)
  const lowerLength = Math.sqrt(lowerArm.x ** 2 + lowerArm.y ** 2 + lowerArm.z ** 2)

  const cosAngle = dot / (upperLength * lowerLength)
  const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)))

  return -(Math.PI - angle)
}

// 頭の回転計算
const calculateHeadRotation = (nose: Landmark, leftEar: Landmark, rightEar: Landmark) => {
  const earCenter = {
    x: (leftEar.x + rightEar.x) / 2,
    y: (leftEar.y + rightEar.y) / 2,
  }

  const yaw = -(nose.x - earCenter.x) * 2
  const pitch = -(nose.y - earCenter.y) * 2

  return { y: yaw, x: pitch }
}

// 表情をVRMに適用
export const applyFaceToVRM = (results: FaceResults, vrm: VRM) => {
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

  blendshapes.forEach((shape: BlendshapeCategory) => {
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
    } catch (e) {}
  }

  const smileLeft = blendshapes.find((s) => s.categoryName === "mouthSmileLeft")?.score || 0
  const smileRight = blendshapes.find((s) => s.categoryName === "mouthSmileRight")?.score || 0
  const smile = Math.max(smileLeft, smileRight)

  if (smile > 0.3 && vrm.expressionManager) {
    try {
      vrm.expressionManager.setValue("happy", smile)
    } catch (e) {}
  }
}
