import time, json
from .modalities import describe_image, transcribe, speak
from .tools import run_agent
from .llm import generate

FORMATS = {
    "LinkedIn post": "a LinkedIn post (120-180 words, hook first line, 3-5 hashtags)",
    "Blog draft": "a blog draft (title, intro, 3 sections with headings, conclusion, ~400 words)",
    "Short summary": "a 3-sentence summary",
    "Tweet thread": "a 4-tweet thread, each under 270 characters",
}


def _timed(timings, name, fn, *a, **k):
    t = time.time()
    out = fn(*a, **k)
    timings[name] = round(time.time() - t, 2)
    return out


def run_pipeline(topic="", image=None, audio=None, audio_mime="audio/wav",
                 fmt="LinkedIn post", tone="professional", voice_out=True):
    timings, steps = {}, {}

    # 1. VOICE IN
    if audio:
        spoken = _timed(timings, "stt", transcribe, audio, audio_mime)
        if not spoken and not topic:
            return {"error": "I couldn't understand the audio. Please re-record in a "
                             "quieter place or type the topic instead."}
        steps["transcript"] = spoken
        topic = f"{topic} {spoken}".strip()

    if not topic and not image:
        return {"error": "Give me a topic, an image, or a voice note."}

    # 2. IMAGE IN
    context = ""
    if image:
        info = _timed(timings, "vision", describe_image, image)
        steps["image_analysis"] = info
        context = ("IMAGE FACTS:\n" + json.dumps(info) +
                   ("\nNOTE: confidence is low; hedge your wording." if info.get("confidence") == "low" else ""))

    # 3. AGENT DRAFT (tool use)
    request = (f"Write {FORMATS[fmt]}. Tone: {tone}. Topic/instruction: "
               f"{topic or 'based on the image'}. Use tools before finalising.")
    agent = _timed(timings, "agent_draft", run_agent, request, context)
    steps["draft"], steps["tool_calls"] = agent["answer"], agent["tool_calls"]

    # 4. SELF-REVIEW
    final = _timed(timings, "polish", generate,
                   f"Improve this {fmt}: fix clarity, remove fluff, keep facts and "
                   f"format. Return only the final text.\n\n{agent['answer']}",
                   temperature=0.4)
    steps["final"] = final

    # 5. VOICE OUT
    if voice_out:
        try:
            steps["audio"] = _timed(timings, "tts", speak, final)
        except Exception as e:
            steps["audio"] = None
            steps["tts_error"] = str(e)

    timings["total"] = round(sum(timings.values()), 2)
    steps["timings"] = timings
    return steps