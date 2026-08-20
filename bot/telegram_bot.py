"""Telegram bot front-end for the hypertension food guidance pipeline.

Understands both English and Arabic automatically (translation happens
inside src/pipeline.py) -- no special handling needed here for language.

Setup (see chat for the fully detailed walkthrough):
  1. Create a bot via @BotFather on Telegram, copy the token it gives you.
  2. Add to .env:  TELEGRAM_BOT_TOKEN=123456789:AA...
  3. Install dependencies:  python -m pip install -r requirements.txt
  4. Run:  python -m bot.telegram_bot
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.generation.labels_ar import ASSESSMENT_CATEGORY_LABELS_AR
from src.pipeline import FoodGuidancePipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "Hi! I'm a food guidance assistant for people with hypertension "
    "(high blood pressure). Ask me about a specific food, e.g.:\n\n"
    "\"Can I eat feta cheese if I have hypertension?\"\n\n"
    "I answer using WHO dietary guidelines, not general knowledge, and "
    "I'm not a substitute for medical advice.\n\n"
    "أهلاً! تقدري تسأليني بالعربي أو الإنجليزي عن أي أكلة."
)


def _format_assessment(assessment, original_language: str) -> str:
    label = assessment.assessment
    if original_language == "ar":
        label = ASSESSMENT_CATEGORY_LABELS_AR.get(label, label)

    why_header = "*ليه:*" if original_language == "ar" else "*Why:*"
    todo_header = "*إيه ممكن تعملي:*" if original_language == "ar" else "*What You Can Do:*"
    evidence_header = "*الدليل:*" if original_language == "ar" else "*Supporting Evidence:*"

    lines = [f"*{assessment.food_name}*", f"_{label}_", "", why_header, assessment.reasoning]

    if assessment.recommendations:
        lines += ["", todo_header] + [f"- {r}" for r in assessment.recommendations]

    if assessment.supporting_evidence:
        lines += ["", evidence_header, assessment.supporting_evidence]

    if assessment.citations:
        lines += ["", "*Sources:*"]
        for c in assessment.citations:
            loc = []
            if c.section_title:
                loc.append(f"Section: {c.section_title}")
            if c.page_start is not None or c.page_end is not None:
                loc.append(f"Page: {c.page_start or '?'}-{c.page_end or '?'}")
            lines.append(f"- {c.document_name} ({', '.join(loc)})")

    lines += ["", f"*Confidence:* {assessment.confidence}"]
    if assessment.generated_by:
        lines.append(f"_Answered by: {assessment.generated_by}_")

    lines += [
        "",
        "This is general dietary information, not medical advice. "
        "Consult your doctor or a registered dietitian.",
    ]

    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_MESSAGE)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = (update.message.text or "").strip()
    if not question:
        return

    pipeline: FoodGuidancePipeline = context.application.bot_data["pipeline"]

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # FoodGuidancePipeline.answer() is slow and blocking (embeddings,
    # reranking, translation, LLM call) -- run it off the event loop so the
    # bot stays responsive to other messages/users while it works.
    result = await asyncio.to_thread(pipeline.answer, question)

    if result.safety_message:
        await update.message.reply_text(result.safety_message)
        return

    if result.confidence_percentage is not None:
        await update.message.reply_text(f"System confidence: {result.confidence_percentage:.1f}%")

    if result.low_confidence_message:
        await update.message.reply_text(result.low_confidence_message)
        return

    if result.assessment:
        text = _format_assessment(result.assessment, result.original_language)
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        return

    await update.message.reply_text("Something went wrong generating a response. Please try again.")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. Add it to your .env file "
            "(get a token from @BotFather on Telegram)."
        )

    application = Application.builder().token(token).build()

    # Build the pipeline ONCE at startup (loads embedding/reranker/
    # translation models, opens the Qdrant connection) and reuse it for
    # every message from every user -- rebuilding it per message would be
    # far too slow.
    logger.info("Loading pipeline (this can take a while on first run)...")
    application.bot_data["pipeline"] = FoodGuidancePipeline()
    logger.info("Pipeline ready.")

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting (polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
