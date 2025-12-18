"use client"

import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import { VRMUtils, VRM } from "@pixiv/three-vrm"
import { PoseLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision"
import { setupThree, setupVRMFromURL, setupMediaPipe, applyPoseToVRM, applyFaceToVRM } from "./lib"

const VRM_MODEL_URL = "https://cdn.glitch.com/29e07830-2317-4b15-a044-135e73c7f840%2FAshtra.vrm"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [status, setStatus] = useState("待機中...")

  // Three.jsとMediaPipeの参照
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const vrmRef = useRef<VRM | null>(null)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null)
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null)
  const isProcessingRef = useRef(false)
  const animationIdRef = useRef<number | null>(null)

  const applyPoseToVRMFunctionRef = useRef(applyPoseToVRM)
  applyPoseToVRMFunctionRef.current = applyPoseToVRM
  
  const applyFaceToVRMFunctionRef = useRef(applyFaceToVRM)
  applyFaceToVRMFunctionRef.current = applyFaceToVRM

  // Three.jsのセットアップ
  const initThreeJS = () => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const { scene, camera, renderer } = setupThree(canvas)
    sceneRef.current = scene
    cameraRef.current = camera
    rendererRef.current = renderer

    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return
      cameraRef.current.aspect = window.innerWidth / window.innerHeight
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(window.innerWidth, window.innerHeight)
    }

    window.addEventListener("resize", handleResize)

    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate)
      if (vrmRef.current) {
        vrmRef.current.update(0.016)
      }
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current)
      }
    }
    animate()

    setStatus("Three.js初期化完了")

    return () => {
      window.removeEventListener("resize", handleResize)
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
    }
  }

  // VRMモデルのロード
  const loadVRMModel = async () => {
    if (!sceneRef.current) return
    try {
      setStatus("VRMモデル読込中...")

      const vrm = await setupVRMFromURL(VRM_MODEL_URL)
      sceneRef.current.add(vrm.scene)
      VRMUtils.rotateVRM0(vrm)
      vrmRef.current = vrm

      setStatus("VRMモデル読込完了")
      console.log("VRM Model loaded:", vrm)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "不明なエラー"
      setStatus("VRMモデル読込エラー: " + errorMessage)
      console.error(error)
    }
  }

  // MediaPipeのセットアップ
  const initMediaPipe = async () => {
    setStatus("MediaPipe初期化中...")
    const { poseLandmarker, faceLandmarker } = await setupMediaPipe()
    poseLandmarkerRef.current = poseLandmarker
    faceLandmarkerRef.current = faceLandmarker

    setStatus("MediaPipe初期化完了")
  }

  // カメラの起動
  const startCamera = async () => {
    const video = videoRef.current
    if (!video) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      })
      video.srcObject = stream
      video.addEventListener("loadeddata", () => {
        setStatus("カメラ起動完了 - 処理開始")
        processFrame()
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "不明なエラー"
      setStatus("カメラエラー: " + errorMessage)
      console.error(error)
    }
  }

  // フレーム処理
  const processFrame = () => {
    const video = videoRef.current
    if (!video || video.readyState !== 4) {
      requestAnimationFrame(processFrame)
      return
    }
    if (isProcessingRef.current) {
      requestAnimationFrame(processFrame)
      return
    }
    isProcessingRef.current = true
    const startTimeMs = performance.now()

    try {
      if (poseLandmarkerRef.current && vrmRef.current) {
        const poseResults = poseLandmarkerRef.current.detectForVideo(video, startTimeMs)
        applyPoseToVRMFunctionRef.current?.(poseResults, vrmRef.current)
      }
      if (faceLandmarkerRef.current && vrmRef.current) {
        const faceResults = faceLandmarkerRef.current.detectForVideo(video, startTimeMs)
        applyFaceToVRMFunctionRef.current?.(faceResults, vrmRef.current)
      }
    } catch (error) {
      console.error("Detection error:", error)
    }
    isProcessingRef.current = false
    requestAnimationFrame(processFrame)
  }

  // 初期化
  useEffect(() => {
    const init = async () => {
      initThreeJS()
      await initMediaPipe()
      await loadVRMModel()
    }

    init()

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
      if (rendererRef.current) {
        rendererRef.current.dispose()
      }
    }
  }, [])

  return (
    <div style={{ margin: 0, overflow: "hidden", fontFamily: "sans-serif" }}>
      <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
          }}
        />
        <video
          ref={videoRef}
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
            onClick={startCamera}
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
