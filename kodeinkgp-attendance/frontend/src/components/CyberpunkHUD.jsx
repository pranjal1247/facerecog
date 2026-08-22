import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useCyberAudio } from '../hooks/useCyberAudio';

export default function CyberpunkHUD() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const wsRef = useRef(null);
  const awaitingResponseRef = useRef(false);
  const captureIntervalRef = useRef(null);
  const prevStateRef = useRef({}); // track_id -> 'recognized' | 'unknown'

  const [tracks, setTracks] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [videoDim, setVideoDim] = useState({ width: 1280, height: 720 });

  const { playLockSound, playAlertSound, getAudioContext } = useCyberAudio();

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error("Camera access denied or unavailable:", err);
      }
    }
    startCamera();
  }, []);

  useEffect(() => {
    function sendFrame(ws) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || ws.readyState !== WebSocket.OPEN) return;
      if (video.videoWidth === 0) return;
      if (awaitingResponseRef.current) return;

      canvas.width = 320;
      canvas.height = 180;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameData = canvas.toDataURL('image/jpeg', 0.5);

      awaitingResponseRef.current = true;
      ws.send(JSON.stringify({
        frame: frameData,
        orig_w: video.videoWidth,
        orig_h: video.videoHeight
      }));
    }

    function connectWS() {
      const ws = new WebSocket("ws://localhost:8000/ws/hud");
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        awaitingResponseRef.current = false;
        captureIntervalRef.current = setInterval(() => sendFrame(ws), 100);
      };

      ws.onmessage = (event) => {
        awaitingResponseRef.current = false;
        try {
          const data = JSON.parse(event.data);
          if (data.frame_width && data.frame_height) {
            setVideoDim({ width: data.frame_width, height: data.frame_height });
          }
          setTracks(data.tracks || []);
        } catch (err) {
          console.error("WS Parse Error:", err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        awaitingResponseRef.current = false;
        if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
        setTimeout(connectWS, 2000);
      };

      ws.onerror = () => ws.close();
    }

    connectWS();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    };
  }, []);

  // Audio reactions: fire once per *transition*, not every frame, using
  // the stable backend track_id so it doesn't spam on every packet.
  useEffect(() => {
    const currentIds = new Set();

    tracks.forEach((t) => {
      currentIds.add(t.track_id);
      const newState = t.is_recognized ? 'recognized' : 'unknown';
      const prevState = prevStateRef.current[t.track_id];

      if (prevState !== newState) {
        if (newState === 'recognized') playLockSound();
        else playAlertSound();
      }
      prevStateRef.current[t.track_id] = newState;
    });

    // Clean up state for tracks that disappeared
    Object.keys(prevStateRef.current).forEach((id) => {
      if (!currentIds.has(Number(id))) delete prevStateRef.current[id];
    });
  }, [tracks, playLockSound, playAlertSound]);

  const getScaledCoords = (bbox) => {
    if (!containerRef.current) return { x: 0, y: 0, w: 0, h: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    const [x1, y1, x2, y2] = bbox;

    const containerAspect = rect.width / rect.height;
    const videoAspect = videoDim.width / videoDim.height;

    let scale, offsetX = 0, offsetY = 0;
    if (videoAspect > containerAspect) {
      scale = rect.height / videoDim.height;
      offsetX = (videoDim.width * scale - rect.width) / 2;
    } else {
      scale = rect.width / videoDim.width;
      offsetY = (videoDim.height * scale - rect.height) / 2;
    }

    return {
      x: x1 * scale - offsetX,
      y: y1 * scale - offsetY,
      w: (x2 - x1) * scale,
      h: (y2 - y1) * scale
    };
  };

  return (
    <div
      ref={containerRef}
      onClick={() => getAudioContext()}
      className="relative w-screen h-screen bg-black overflow-hidden select-none font-mono text-cyan-400"
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover opacity-80"
      />
      <canvas ref={canvasRef} className="hidden" />

      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-transparent via-black/40 to-black pointer-events-none" />
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#00f0ff08_1px,transparent_1px),linear-gradient(to_bottom,#00f0ff08_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none" />

      <header className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center z-30 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center gap-2 border border-cyan-500/40 px-3 py-1 bg-black/60 backdrop-blur">
          <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
          <span className="text-xs tracking-widest text-cyan-300">
            {wsConnected ? 'SYSTEM ONLINE // WS ACTIVE' : 'DISCONNECTED // RECONNECTING'}
          </span>
        </div>
        <div className="text-xs text-cyan-500">TARGETS: {tracks.length}</div>
      </header>

      <div className="absolute inset-0 z-20 pointer-events-none">
        <AnimatePresence>
          {tracks.map((track) => {
            const coords = getScaledCoords(track.bbox);
            const themeColor = track.is_recognized ? '#00ff99' : '#ff9900';
            const profile = track.profile;

            // Flip the card to the left if it'd overflow the right edge
            const containerWidth = containerRef.current?.clientWidth || 1920;
            const flipLeft = coords.x + coords.w + 260 > containerWidth;

            return (
              <motion.div
                key={track.track_id}
                initial={{ opacity: 0 }}
                animate={{
                  opacity: 1,
                  left: coords.x,
                  top: coords.y,
                  width: coords.w,
                  height: coords.h,
                }}
                exit={{ opacity: 0 }}
                transition={{
                  left: { type: "spring", stiffness: 300, damping: 30 },
                  top: { type: "spring", stiffness: 300, damping: 30 },
                  width: { type: "spring", stiffness: 300, damping: 30 },
                  height: { type: "spring", stiffness: 300, damping: 30 },
                  opacity: { duration: 0.15 },
                }}
                style={{
                  position: 'absolute',
                  borderColor: themeColor
                }}
                className="border-2"
              >
                <div style={{ backgroundColor: themeColor }} className="absolute -top-7 left-0 px-2 py-0.5 text-black font-bold text-xs">
                  {profile?.name || 'UNKNOWN'}
                </div>

                {profile && (
                  <motion.div
                    initial={{ opacity: 0, x: flipLeft ? 10 : -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                    style={{
                      position: 'absolute',
                      top: 0,
                      [flipLeft ? 'right' : 'left']: flipLeft ? 'calc(100% + 12px)' : 'calc(100% + 12px)',
                      borderColor: themeColor,
                    }}
                    className="w-56 bg-black/80 backdrop-blur border px-3 py-2 text-xs space-y-1 shadow-[0_0_15px_rgba(0,255,153,0.3)]"
                  >
                    <div className="text-emerald-400 font-bold tracking-widest text-sm">{profile.name}</div>
                    <div className="text-cyan-300">{profile.department}</div>
                    {profile.hall && <div className="text-cyan-300">🏠 {profile.hall}</div>}
                    {profile.year && <div className="text-cyan-300">🎓 {profile.year}</div>}
                    {profile.fun_fact && (
                      <div className="text-amber-300 italic pt-1 border-t border-cyan-500/20">
                        "{profile.fun_fact}"
                      </div>
                    )}
                  </motion.div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* KodeinKGP logo, bottom-left, no background */}
      <img
        src="/kodeinkgp-logo.png"
        alt="KodeinKGP"
        className="absolute bottom-4 left-4 w-52 opacity-90 pointer-events-none z-30 drop-shadow-[0_0_8px_rgba(0,240,255,0.5)]"
      />
    </div>
  );
}