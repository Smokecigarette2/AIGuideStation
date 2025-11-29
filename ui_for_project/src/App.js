import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function splitIntoChunks(text, maxLen = 220) {
  const clean = (text || "").trim();
  if (!clean) return [];

  const parts = clean
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?…])\s+/g);

  const chunks = [];
  let buf = "";

  for (const p of parts) {
    if ((buf + " " + p).trim().length <= maxLen) {
      buf = (buf ? buf + " " : "") + p;
    } else {
      if (buf) chunks.push(buf);

      if (p.length > maxLen) {
        for (let i = 0; i < p.length; i += maxLen)
          chunks.push(p.slice(i, i + maxLen));
        buf = "";
      } else {
        buf = p;
      }
    }
  }

  if (buf) chunks.push(buf);
  return chunks;
}

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const [listening, setListening] = useState(false);

  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [speaking, setSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState("");

  const recognitionRef = useRef(null);
  const questionRef = useRef("");
  const pendingVoiceQuestionRef = useRef("");

  const ttsQueueRef = useRef([]);
  const ttsIndexRef = useRef(0);

  const inFlight = useRef(false);

  const ORG_ID = "my_university";

// eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    questionRef.current = question;
  }, [question]);

// eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn("SpeechRecognition не поддерживается.");
      return;
    }

    const rec = new SpeechRecognition();
    rec.lang = "ru-RU";
    rec.interimResults = false;
    rec.maxAlternatives = 1;

    rec.onresult = (event) => {
      const text = event.results[0][0].transcript.trim();
      const combined = (questionRef.current ? questionRef.current + " " : "") + text;

      pendingVoiceQuestionRef.current = combined;
      setQuestion(combined);
    };

    rec.onend = () => {
      setListening(false);

      const q = pendingVoiceQuestionRef.current.trim();
      if (q && !inFlight.current) {
        pendingVoiceQuestionRef.current = "";
        askServer(q);
      }
    };

    rec.onerror = (e) => {
      console.error("SpeechRecognition error:", e);
      setListening(false);
    };

    recognitionRef.current = rec;

    return () => rec.abort();
  }, []);

  const toggleListening = () => {
    const rec = recognitionRef.current;
    if (!rec) return alert("Ваш браузер не поддерживает голосовой ввод.");

    if (!listening) {
      stopTTS();
      pendingVoiceQuestionRef.current = "";
      setListening(true);
      rec.start();
    } else {
      rec.stop();
    }
  };

// eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!("speechSynthesis" in window)) {
      console.warn("speechSynthesis не поддерживается.");
      return;
    }

    const loadVoices = () => {
      const v = window.speechSynthesis.getVoices() || [];
      setVoices(v);

      if (!selectedVoiceURI) {
        const ru = v.find((x) => x.lang?.toLowerCase().startsWith("ru"));
        if (ru) setSelectedVoiceURI(ru.voiceURI);
      }
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const stopTTS = () => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
    ttsQueueRef.current = [];
    ttsIndexRef.current = 0;
  };

  const speakText = (text) => {
    if (!("speechSynthesis" in window)) return;

    stopTTS();

    const chunks = splitIntoChunks(text, 220);
    if (!chunks.length) return;

    ttsQueueRef.current = chunks;
    ttsIndexRef.current = 0;

    const playNext = () => {
      const idx = ttsIndexRef.current;
      const queue = ttsQueueRef.current;

      if (idx >= queue.length) {
        setSpeaking(false);
        return;
      }

      const u = new SpeechSynthesisUtterance(queue[idx]);
      u.lang = "ru-RU";

      const voice =
        voices.find((v) => v.voiceURI === selectedVoiceURI) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("ru")) ||
        null;

      if (voice) u.voice = voice;

      u.rate = 1;
      u.pitch = 1;

      u.onstart = () => setSpeaking(true);
      u.onend = () => {
        ttsIndexRef.current += 1;
        playNext();
      };
      u.onerror = (e) => {
        console.error("TTS error:", e);
        setSpeaking(false);
      };

      window.speechSynthesis.speak(u);
    };

    playNext();
  };
// eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (answer && ttsEnabled) speakText(answer);
  }, [answer]);

  // ==========================
  //  askServer()
  // ==========================
  const askServer = async (override) => {
    const q = (override ?? questionRef.current).trim();
    if (!q) return;

    if (inFlight.current) return;
    inFlight.current = true;

    setLoading(true);
    setAnswer("");
    stopTTS();

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          org_id: ORG_ID,
          question: q,
          language: "ru",
          top_k: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setAnswer(data?.detail ? String(data.detail) : "Ошибка сервера");
      } else {
        setAnswer(data.answer_text || "Пустой ответ");
      }
    } catch (err) {
      console.error(err);
      setAnswer("Не удалось подключиться к серверу");
    }

    setLoading(false);
    inFlight.current = false;
  };

  // ==========================
  // UI
  // ==========================
  return (
    <div className="container">
      <h1 className="title">AI Guide Station</h1>

      <div className="input-row">
        <textarea
          className="input"
          placeholder="Введите вопрос..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

        <button
          type="button"
          className={`mic-button ${listening ? "mic-button--active" : ""}`}
          onClick={toggleListening}
          title="Голосовой ввод"
        >
          {listening ? "🎙" : "🎤"}
        </button>
      </div>

      <div className="tts-row">
        <label className="tts-toggle">
          <input
            type="checkbox"
            checked={ttsEnabled}
            onChange={(e) => setTtsEnabled(e.target.checked)}
          />
          <span>Озвучивать ответы</span>
        </label>

        <select
          className="tts-select"
          value={selectedVoiceURI}
          onChange={(e) => setSelectedVoiceURI(e.target.value)}
          disabled={!voices.length}
        >
          {voices.length === 0 ? (
            <option>Голоса не найдены</option>
          ) : (
            voices.map((v) => (
              <option key={v.voiceURI} value={v.voiceURI}>
                {v.name} ({v.lang})
              </option>
            ))
          )}
        </select>

        <button
          className="tts-button"
          onClick={() => speakText(answer)}
          disabled={!answer}
        >
          🔊
        </button>

        <button
          className="tts-button"
          onClick={stopTTS}
          disabled={!speaking}
        >
          ⏹
        </button>
      </div>

      <button className="button" onClick={() => askServer()} disabled={loading}>
        {loading ? "Загрузка..." : "Отправить"}
      </button>

      {answer && (
        <div className="answer-box">
          <h2>Ответ:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;