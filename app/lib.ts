import * as THREE from "three"
import {
  FilesetResolver,
  PoseLandmarker,
  FaceLandmarker,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision"
import { VRMLoaderPlugin, VRM, VRMExpressionPresetName } from "@pixiv/three-vrm"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader"
import * as Kalidokit from "kalidokit"

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

// ポーズをVRMに適用（Kalidokit使用）
export const applyPoseToVRM = (
  results: { landmarks: NormalizedLandmark[][], worldLandmarks?: NormalizedLandmark[][] },
  vrm: VRM
) => {
  if (!results.landmarks || results.landmarks.length === 0) return
  if (!vrm) return

  const landmarks = results.landmarks[0]
  const worldLandmarks = results.worldLandmarks?.[0] || landmarks

  // Kalidokitを使って姿勢を計算
  const riggedPose = Kalidokit.Pose.solve(worldLandmarks, landmarks, {
    runtime: "mediapipe",
    video: undefined as any,
  })

  if (!riggedPose) return

  const humanoid = vrm.humanoid

  // 上半身の回転を適用
  const spine = humanoid.getNormalizedBoneNode("spine")
  if (spine && riggedPose.Spine) {
    rigRotation(spine, riggedPose.Spine, 0.25)
  }

  // 胸部の回転を適用
  const chest = humanoid.getNormalizedBoneNode("chest")
  if (chest && riggedPose.Spine) {
    rigRotation(chest, riggedPose.Spine, 0.25)
  }

  // 首の回転を適用
  const neck = humanoid.getNormalizedBoneNode("neck")
  if (neck && riggedPose.Spine) {
    rigRotation(neck, riggedPose.Spine, 0.25)
  }

  // 頭部の回転を適用
  const head = humanoid.getNormalizedBoneNode("head")
  if (head && riggedPose.Hips?.rotation) {
    rigRotation(head, riggedPose.Hips.rotation, 1, 0.3)
  }

  // 腰の回転を適用
  const hips = humanoid.getNormalizedBoneNode("hips")
  if (hips && riggedPose.Hips?.rotation) {
    rigRotation(hips, riggedPose.Hips.rotation, 0.25)
  }

  // 左腕の回転を適用
  const leftUpperArm = humanoid.getNormalizedBoneNode("leftUpperArm")
  if (leftUpperArm && riggedPose.LeftUpperArm) {
    rigRotation(leftUpperArm, riggedPose.LeftUpperArm, 1, 0.3)
  }

  const leftLowerArm = humanoid.getNormalizedBoneNode("leftLowerArm")
  if (leftLowerArm && riggedPose.LeftLowerArm) {
    rigRotation(leftLowerArm, riggedPose.LeftLowerArm, 1, 0.3)
  }

  // 右腕の回転を適用
  const rightUpperArm = humanoid.getNormalizedBoneNode("rightUpperArm")
  if (rightUpperArm && riggedPose.RightUpperArm) {
    rigRotation(rightUpperArm, riggedPose.RightUpperArm, 1, 0.3)
  }

  const rightLowerArm = humanoid.getNormalizedBoneNode("rightLowerArm")
  if (rightLowerArm && riggedPose.RightLowerArm) {
    rigRotation(rightLowerArm, riggedPose.RightLowerArm, 1, 0.3)
  }

  // 左脚の回転を適用
  const leftUpperLeg = humanoid.getNormalizedBoneNode("leftUpperLeg")
  if (leftUpperLeg && riggedPose.LeftUpperLeg) {
    rigRotation(leftUpperLeg, riggedPose.LeftUpperLeg, 1, 0.3)
  }

  const leftLowerLeg = humanoid.getNormalizedBoneNode("leftLowerLeg")
  if (leftLowerLeg && riggedPose.LeftLowerLeg) {
    rigRotation(leftLowerLeg, riggedPose.LeftLowerLeg, 1, 0.3)
  }

  // 右脚の回転を適用
  const rightUpperLeg = humanoid.getNormalizedBoneNode("rightUpperLeg")
  if (rightUpperLeg && riggedPose.RightUpperLeg) {
    rigRotation(rightUpperLeg, riggedPose.RightUpperLeg, 1, 0.3)
  }

  const rightLowerLeg = humanoid.getNormalizedBoneNode("rightLowerLeg")
  if (rightLowerLeg && riggedPose.RightLowerLeg) {
    rigRotation(rightLowerLeg, riggedPose.RightLowerLeg, 1, 0.3)
  }
}

// 回転を適用するヘルパー関数
const rigRotation = (
  bone: THREE.Object3D,
  rotation: { x: number; y: number; z: number },
  dampener = 1,
  lerpAmount = 0.3
) => {
  if (!rotation) return
  
  const euler = new THREE.Euler(
    rotation.x * dampener,
    rotation.y * dampener,
    rotation.z * dampener
  )
  const quaternion = new THREE.Quaternion().setFromEuler(euler)
  bone.quaternion.slerp(quaternion, lerpAmount)
}

// 表情をVRMに適用（Kalidokit使用）
export const applyFaceToVRM = (results: {
  faceBlendshapes: {
    categories: {
      categoryName: string
      score: number
    }[]
  }[]
  faceLandmarks?: NormalizedLandmark[][]
}, vrm: VRM) => {
  if (!results.faceBlendshapes || results.faceBlendshapes.length === 0) return
  if (!vrm.expressionManager) return

  const blendshapes = results.faceBlendshapes[0].categories
  const faceLandmarks = results.faceLandmarks?.[0]

  // Kalidokitで顔の姿勢を計算
  if (faceLandmarks) {
    const riggedFace = Kalidokit.Face.solve(faceLandmarks, {
      runtime: "mediapipe",
      video: undefined as any,
    })

    if (riggedFace) {
      // 頭部の回転を適用
      const head = vrm.humanoid.getNormalizedBoneNode("head")
      if (head && riggedFace.head) {
        rigRotation(head, riggedFace.head, 1, 0.7)
      }

      // 瞳の回転
      // if (riggedFace.pupil) {
      //   const leftEye = vrm.humanoid.getNormalizedBoneNode("leftEye")
      //   const rightEye = vrm.humanoid.getNormalizedBoneNode("rightEye")
        
      //   // 左右の瞳が追従
      //   if (leftEye) {
      //     leftEye.rotation.y = riggedFace.pupil.x
      //     leftEye.rotation.z = riggedFace.pupil.y
      //   }
      //   if (rightEye) {
      //     rightEye.rotation.y = riggedFace.pupil.x
      //     rightEye.rotation.z = riggedFace.pupil.y
      //   }
      // }
    }
  }

  // 表情のブレンドシェイプを適用
  const blendshape_map: Record<string, VRMExpressionPresetName | string> = {
    eyeBlinkLeft: "blinkLeft",
    eyeBlinkRight: "blinkRight",
    jawOpen: "aa",
    mouthSmileLeft: "happy",
    mouthSmileRight: "happy",
    browDownLeft: "angry",
    browDownRight: "angry",
    mouthFrownLeft: "sad",
    mouthFrownRight: "sad",
  }

  // すべての表情をリセット
  vrm.expressionManager.setValue("aa", 0)
  vrm.expressionManager.setValue("ih", 0)
  vrm.expressionManager.setValue("ou", 0)
  vrm.expressionManager.setValue("ee", 0)
  vrm.expressionManager.setValue("oh", 0)
  vrm.expressionManager.setValue("blink", 0)
  vrm.expressionManager.setValue("blinkLeft", 0)
  vrm.expressionManager.setValue("blinkRight", 0)
  vrm.expressionManager.setValue("happy", 0)
  vrm.expressionManager.setValue("angry", 0)
  vrm.expressionManager.setValue("sad", 0)
  vrm.expressionManager.setValue("relaxed", 0)

  blendshapes.forEach((shape) => {
    const vrm_expression = blendshape_map[shape.categoryName]
    if (vrm_expression && shape.score > 0.1 && vrm.expressionManager) {
      try {
        vrm.expressionManager.setValue(vrm_expression as VRMExpressionPresetName, shape.score)
      } catch (e) {
        // 対応していない表情は無視
      }
    }
  })

  // まばたき処理
  const blink_left = blendshapes.find((s) => s.categoryName === "eyeBlinkLeft")?.score || 0
  const blink_right = blendshapes.find((s) => s.categoryName === "eyeBlinkRight")?.score || 0
  
  if (blink_left > 0.5) {
    vrm.expressionManager.setValue("blinkLeft", blink_left)
  }
  if (blink_right > 0.5) {
    vrm.expressionManager.setValue("blinkRight", blink_right)
  }

  const blink = (blink_left + blink_right) / 2
  if (blink > 0.6) {
    vrm.expressionManager.setValue("blink", blink)
  }

  // 笑顔処理
  const smile_left = blendshapes.find((s) => s.categoryName === "mouthSmileLeft")?.score || 0
  const smile_right = blendshapes.find((s) => s.categoryName === "mouthSmileRight")?.score || 0
  const smile = Math.max(smile_left, smile_right)

  if (smile > 0.3) {
    vrm.expressionManager.setValue("happy", smile)
  }

  // 口の開き
  const mouth_open = blendshapes.find((s) => s.categoryName === "jawOpen")?.score || 0
  if (mouth_open > 0.3) {
    vrm.expressionManager.setValue("aa", mouth_open)
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
