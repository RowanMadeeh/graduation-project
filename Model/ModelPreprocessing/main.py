import io
from pipes import quote
from typing import IO, Iterator

from fastapi import FastAPI, UploadFile, Request, File, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from SummarizationPipeline import SummarizationService
from Events import SSEEvent
import asyncio
from sse_starlette import EventSourceResponse
app = FastAPI()
summarizationService = SummarizationService()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# @app.post("/full-text")
# async def get_full_text(file: UploadFile):
#     ocr_service = OCRService()
#     text = await ocr_service.process_pdf(file)
#     return {"text": text}



# from ChunkingService import ChunkingService
# @app.post("/chunk-spacy")
# async def get_chunk_sentences_spacy(file: UploadFile):
#     ocr_service = OCRService()
#     chunking = ChunkingService()
#     text = await ocr_service.process_pdf(file)
#     temp = await chunking.semantic_chunk_using_spacy(text)
#     response = {}
#     for i in range(len(temp)):
#         response[i] = temp[i]
#     return response



# @app.post("/overlappingChunking")
# async def getOverlappingChunks(file : UploadFile):
#     text = await ocr_service.process_pdf(file)
#     temp = await chunking.overlapping_chunking(text)
#     response = {}
#     for i in range(len(temp)):
#         response[i] = temp[i]
#     return response



"""
    Endpoint to get Summary,
    It takes a file as an input and returns array of bytes representing the summary pdf
"""
@app.post("/getSummary")
async def getSummary(file: UploadFile = File(...)):
    summary = await summarizationService.run(file)


    return StreamingResponse(
        io.BytesIO(summary),
        media_type="application/pdf",
        headers={
            # This header will make Spring’s WebClient (or a browser) see it as a downloadable file
            "Content-Disposition": 'attachment; filename="book.pdf"'
        }
    )



"""
    Endpoint used to keep the user updated with the current progress of the pipeline
"""
@app.get("/stream")
async def stream_events(req: Request):
    async def stream_generator():
        while True:
            event = SSEEvent.get_event()
            if event:
                temp = event.model_dump()
                if temp['message'] == 'end':
                    break
                else:
                    yield "data: {}\n\n".format(event.model_dump_json())

            await asyncio.sleep(0.01)

    return EventSourceResponse(stream_generator())

