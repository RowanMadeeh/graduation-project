import time

from transformers import AutoTokenizer
import asyncio
from AbstractiveService import AbstractiveService
from ChunkingService import ChunkingService
from ExtractiveService import ExtractiveService
from OCRService import OCRService
from fastapi import UploadFile
from Events import SSEEvent, EventModel
from SmoothingService import SmoothingService


class SummarizationService:
    def __init__(self):
        self.ocr_service = OCRService()
        self.chunking_service = ChunkingService()
        self.extractiveService = ExtractiveService()
        self.abstractiveService = AbstractiveService()
        self.smoothingService = SmoothingService()
        self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
        self.MAX_TOKENS_PER_CHUNK = 700


    def get_tokens(self,text):
        tokens = self.tokenizer(text, return_tensors="pt").input_ids
        token_count = tokens.shape[1]
        return token_count





    SSEEvent.add_event(EventModel(message="الملخص جاهز! ", percentage=100))

    async def run(self, file : UploadFile) -> bytes:
        start_total = time.perf_counter()
        SSEEvent.add_event(EventModel(message="جارٍ قراءة الملف...", percentage=0))
        await asyncio.sleep(1)
        fullText = await self.ocr_service.process_pdf(file)
        SSEEvent.add_event(EventModel(message="تمت قراءة الملف بنجاح", percentage=10))
        SSEEvent.add_event(EventModel(message="جارٍ تجهيز المحتوى للمعالجة...", percentage=10))
        await asyncio.sleep(1)
        chunks = await self.chunking_service.semantic_chunk_using_spacy(fullText)
        SSEEvent.add_event(EventModel(message="تم تجهيز المحتوى", percentage=20))
        SSEEvent.add_event(EventModel(message="جارٍ تلخيص النصوص الأساسية...", percentage=20))
        await asyncio.sleep(1)

        """
             Extractive
        """
        extractiveSummary = []
        for chunk in chunks:
            extractiveSummary.append(self.extractiveService.getSummary(chunk) + "\n")
        SSEEvent.add_event(EventModel(message="تم تلخيص النصوص الأساسية", percentage=40))
        SSEEvent.add_event(EventModel(message="جارٍ إنشاء ملخص نهائي...", percentage=40))
        await asyncio.sleep(1)
        cur = ""
        cnt = 1
        abstractiveSummary = []

        # For testing
        for chunk in extractiveSummary:
            if self.get_tokens(cur) + self.get_tokens(chunk) > 1024:
                cnt += 1
                abstractiveSummary.append(cur)
                cur = chunk
            else:
                cur += chunk
        if cur:
            abstractiveSummary.append(cur)
            cnt += 1
        print(cnt)
        # Until here

        """
             Abstractive
        """
        finalSummary = []
        cnt = 1
        startTime = time.perf_counter()

        cur = ""
        for chunk in abstractiveSummary:
            print("Processing Chunk " + str(cnt))
            cnt += 1
            temp = self.abstractiveService.generate_summary(chunk)
            if self.get_tokens(cur) + self.get_tokens(temp) > self.MAX_TOKENS_PER_CHUNK:
                finalSummary.append(cur)
                cur = temp
            else:
                cur += temp
        if cur:
            finalSummary.append(cur)
        endTime = time.perf_counter()
        SSEEvent.add_event(EventModel(message="تم إنشاء الملخص النهائي", percentage=90))
        print("Abstractive Summary Time " + str(endTime - startTime))

        """
             Smoothing
        """
        SSEEvent.add_event(EventModel(message="جارٍ تحسين النتائج النهائية...", percentage=90))
        await asyncio.sleep(1)
        ret = self.smoothingService.process_text_file(finalSummary)
        SSEEvent.add_event(EventModel(message="تم تحسين النتائج النهائية", percentage=90))
        await asyncio.sleep(1)
        SSEEvent.add_event(EventModel(message="الملخص جاهز! ", percentage=100))
        await asyncio.sleep(1)
        end_total = time.perf_counter()
        print("Total Time " + str(end_total - start_total))
        return ret
