"use client"

import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import { VRMUtils, VRM } from "@pixiv/three-vrm"
import { PoseLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision"
import { setupThree, setupVRMFromURL, setupMediaPipe, applyPoseToVRM, applyFaceToVRM } from "./lib"

const VRM_MODEL_URL = "https://cdn.glitch.com/29e07830-2317-4b15-a044-135e73c7f840%2FAshtra.vrm"

export default function Home() {
  const canvas_ref = useRef<HTMLCanvasElement>(null)
  const video_ref = useRef<HTMLVideoElement>(null)
  const [status, set_status] = useState("待機中...")

  // Three.jsとMediaPipeの参照
  const scene_ref = useRef<THREE.Scene | null>(null)
  const camera_ref = useRef<THREE.PerspectiveCamera | null>(null)
  const renderer_ref = useRef<THREE.WebGLRenderer | null>(null)
  const vrm_ref = useRef<VRM | null>(null)
  const pose_landmarker_ref = useRef<PoseLandmarker | null>(null)
  const face_landmarker_ref = useRef<FaceLandmarker | null>(null)
  const is_processing_ref = useRef(false)
  const animation_id_ref = useRef<number | null>(null)

  const apply_pose_to_vrm_function_ref = useRef(applyPoseToVRM)
  apply_pose_to_vrm_function_ref.current = applyPoseToVRM
  
  const apply_face_to_vrm_function_ref = useRef(applyFaceToVRM)
  apply_face_to_vrm_function_ref.current = applyFaceToVRM

  // Three.jsのセットアップ
  const init_three_js = () => {
    if (!canvas_ref.current) return

    const canvas = canvas_ref.current
    const { scene, camera, renderer } = setupThree(canvas)
    scene_ref.current = scene
    camera_ref.current = camera
    renderer_ref.current = renderer

    const handle_resize = () => {
      if (!camera_ref.current || !renderer_ref.current) return
      camera_ref.current.aspect = window.innerWidth / window.innerHeight
      camera_ref.current.updateProjectionMatrix()
      renderer_ref.current.setSize(window.innerWidth, window.innerHeight)
    }

    window.addEventListener("resize", handle_resize)

    const animate = () => {
      animation_id_ref.current = requestAnimationFrame(animate)
      if (vrm_ref.current) {
        vrm_ref.current.update(0.016)
      }
      if (renderer_ref.current && scene_ref.current && camera_ref.current) {
        renderer_ref.current.render(scene_ref.current, camera_ref.current)
      }
    }
    animate()

    set_status("Three.js初期化完了")

    return () => {
      window.removeEventListener("resize", handle_resize)
      if (animation_id_ref.current) {
        cancelAnimationFrame(animation_id_ref.current)
      }
    }
  }

  // VRMモデルのロード
  const load_vrm_model = async () => {
    if (!scene_ref.current) return
    try {
      set_status("VRMモデル読込中...")

      const vrm = await setupVRMFromURL(VRM_MODEL_URL)
      scene_ref.current.add(vrm.scene)
      VRMUtils.rotateVRM0(vrm)
      vrm_ref.current = vrm

      set_status("VRMモデル読込完了")
      console.log("VRM Model loaded:", vrm)
    } catch (error) {
      const error_message = error instanceof Error ? error.message : "不明なエラー"
      set_status("VRMモデル読込エラー: " + error_message)
      console.error(error)
    }
  }

  // MediaPipeのセットアップ
  const init_media_pipe = async () => {
    set_status("MediaPipe初期化中...")
    const { poseLandmarker, faceLandmarker } = await setupMediaPipe()
    pose_landmarker_ref.current = poseLandmarker
    face_landmarker_ref.current = faceLandmarker

    set_status("MediaPipe初期化完了")
  }

  // カメラの起動
  const start_camera = async () => {
    const video = video_ref.current
    if (!video) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      })
      video.srcObject = stream
      video.addEventListener("loadeddata", () => {
        set_status("カメラ起動完了 - 処理開始")
        process_frame()
      })
    } catch (error) {
      const error_message = error instanceof Error ? error.message : "不明なエラー"
      set_status("カメラエラー: " + error_message)
      console.error(error)
    }
  }

  // フレーム処理
  const process_frame = () => {
    const video = video_ref.current
    if (!video || video.readyState !== 4) {
      requestAnimationFrame(process_frame)
      return
    }
    if (is_processing_ref.current) {
      requestAnimationFrame(process_frame)
      return
    }
    is_processing_ref.current = true
    const start_time_ms = performance.now()

    try {
      if (pose_landmarker_ref.current && vrm_ref.current) {
        const pose_results = pose_landmarker_ref.current.detectForVideo(video, start_time_ms)
        apply_pose_to_vrm_function_ref.current?.(pose_results, vrm_ref.current)
      }
      if (face_landmarker_ref.current && vrm_ref.current) {
        const face_results = face_landmarker_ref.current.detectForVideo(video, start_time_ms)
        apply_face_to_vrm_function_ref.current?.(face_results, vrm_ref.current)
      }
    } catch (error) {
      console.error("Detection error:", error)
    }
    is_processing_ref.current = false
    requestAnimationFrame(process_frame)
  }

  // 初期化
  useEffect(() => {
    const init = async () => {
      init_three_js()
      await init_media_pipe()
      await load_vrm_model()
    }

    init()

    return () => {
      if (animation_id_ref.current) {
        cancelAnimationFrame(animation_id_ref.current)
      }
      if (renderer_ref.current) {
        renderer_ref.current.dispose()
      }
    }
  }, [])

  return (
    <div style={{ margin: 0, overflow: "hidden", fontFamily: "sans-serif" }}>
      <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
        <canvas
          ref={canvas_ref}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
          }}
        />
        <video
          ref={video_ref}
          autoPlay
          playsInline
          style={{
            position: "absolute",
            top: "10px",
            left: "10px",
            width: "320px",
            height: "240px",
            border: "2px solid #fff",
            borderRadius: "8px",
            transform: "scaleX(-1)",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: "10px",
            right: "10px",
            background: "rgba(0, 0, 0, 0.7)",
            color: "white",
            padding: "15px",
            borderRadius: "8px",
            fontSize: "12px",
          }}
        >
          <button
            onClick={start_camera}
            style={{
              margin: "5px 0",
              padding: "8px 12px",
              cursor: "pointer",
              background: "#4CAF50",
              color: "white",
              border: "none",
              borderRadius: "4px",
            }}
          >
            カメラ開始
          </button>
          <div
            style={{
              marginTop: "10px",
              fontSize: "11px",
              color: "#ffeb3b",
            }}
          >
            {status}
          </div>
        </div>
      </div>
    </div>
  )
}
