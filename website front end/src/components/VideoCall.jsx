// src/VideoCall.jsx
import React, { useEffect, useRef, useState } from "react";
import Peer from "peerjs";
import { useLocation, useNavigate } from "react-router-dom";

function VideoCall() {
  const [myId, setMyId] = useState("");
  const [remoteId, setRemoteId] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);
  const [isDoctor, setIsDoctor] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerRef = useRef(null);
  const currentCall = useRef(null);
  const streamRef = useRef(null);
  
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is doctor or patient from URL params
    const params = new URLSearchParams(location.search);
    const role = params.get("role");
    setIsDoctor(role === "doctor");

    const peer = new Peer();
    peerRef.current = peer;

    peer.on("open", (id) => {
      setMyId(id);
      setLoading(false);
    });

    peer.on("error", (err) => {
      setError("Connection error: " + err.message);
      setLoading(false);
    });

    navigator.mediaDevices
      .getUserMedia({ video: true, audio: true })
      .then((stream) => {
        streamRef.current = stream;
        if (localVideoRef.current) {
          localVideoRef.current.srcObject = stream;
        }

        peer.on("call", (call) => {
          call.answer(stream);
          currentCall.current = call;
          handleCallStream(call);
        });
      })
      .catch((err) => {
        setError("Camera/Microphone error: " + err.message);
        setLoading(false);
      });

    return () => {
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (currentCall.current) {
        currentCall.current.close();
      }
      if (peerRef.current) {
        peerRef.current.destroy();
      }
    };
  }, [location]);

  const handleCallStream = (call) => {
    call.on("stream", (remoteStream) => {
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = remoteStream;
      }
    });

    call.on("close", () => {
      setIsConnected(false);
    });
  };

  const startCall = () => {
    if (!remoteId.trim()) {
      setError("Please enter remote ID");
      return;
    }

    try {
      const call = peerRef.current.call(remoteId, streamRef.current);
      currentCall.current = call;
      handleCallStream(call);
      setIsConnected(true);
      setError(null);
    } catch (err) {
      setError("Failed to start call: " + err.message);
    }
  };

  const endCall = () => {
    if (currentCall.current) {
      currentCall.current.close();
      setIsConnected(false);
    }
  };

  const toggleMute = () => {
    if (streamRef.current) {
      const audioTrack = streamRef.current.getAudioTracks()[0];
      audioTrack.enabled = !audioTrack.enabled;
      setIsMuted(!audioTrack.enabled);
    }
  };

  const toggleVideo = () => {
    if (streamRef.current) {
      const videoTrack = streamRef.current.getVideoTracks()[0];
      videoTrack.enabled = !videoTrack.enabled;
      setIsVideoOff(!videoTrack.enabled);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">جاري تحميل الاتصال المرئي...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-6xl mx-auto bg-white rounded-xl shadow-lg p-6">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            {isDoctor ? "مكالمة مرئية مع المريض" : "مكالمة مرئية مع الدكتور"}
          </h2>
          {error && (
            <div className="bg-red-50 text-red-600 p-3 rounded-lg mb-4">
              {error}
            </div>
          )}
          <div className="bg-gray-50 p-3 rounded-lg inline-block">
            <p className="text-gray-600">
              <span className="font-semibold">معرف الاتصال الخاص بك:</span>{" "}
              {myId}
            </p>
          </div>
        </div>

        {!isConnected && (
          <div className="flex justify-center gap-4 mb-6">
            <input
              type="text"
              placeholder="أدخل معرف المستخدم الآخر"
              value={remoteId}
              onChange={(e) => setRemoteId(e.target.value)}
              className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              onClick={startCall}
              className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
            >
              بدء المكالمة
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="relative">
            <h4 className="text-lg font-semibold mb-2 text-center">
              {isDoctor ? "أنت (الدكتور)" : "أنت (المريض)"}
            </h4>
            <video
              ref={localVideoRef}
              muted
              autoPlay
              playsInline
              className="w-full rounded-lg border border-gray-200"
            />
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2">
              <button
                onClick={toggleMute}
                className={`p-2 rounded-full ${
                  isMuted ? "bg-red-500" : "bg-gray-800"
                } text-white`}
              >
                {isMuted ? "🔇" : "🎤"}
              </button>
              <button
                onClick={toggleVideo}
                className={`p-2 rounded-full ${
                  isVideoOff ? "bg-red-500" : "bg-gray-800"
                } text-white`}
              >
                {isVideoOff ? "🚫" : "📹"}
              </button>
            </div>
          </div>

          <div className="relative">
            <h4 className="text-lg font-semibold mb-2 text-center">
              {isDoctor ? "المريض" : "الدكتور"}
            </h4>
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full rounded-lg border border-gray-200"
            />
          </div>
        </div>

        {isConnected && (
          <div className="text-center mt-6">
            <button
              onClick={endCall}
              className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition-colors"
            >
              إنهاء المكالمة
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoCall;
