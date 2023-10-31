"""
A model worker that executes the model based on Sentence-Transformers for BERT.

"""

import argparse
import asyncio
import numpy
import base64

from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.model_worker import (
    BaseModelWorker,
    logger,
    worker_id,
)
from sentence_transformers import SentenceTransformer
import torch


app = FastAPI()

class BERTWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: BERT worker..."
        )
        self.model = SentenceTransformer(model_name_or_path=model_path)
        self.context_len = self.model.get_max_seq_length

        if not no_register:
            self.init_heart_beat()

    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            ret = {"embedding": [], "token_num": 0}
            sentence_embeddings = self.model.encode(sentences=params["input"],show_progress_bar=False)
            base64_encode = params.get("encoding_format", None)

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(sentence_embeddings)
            else:
                out_embeddings = sentence_embeddings.tolist()
            ret["embedding"] = out_embeddings

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    def __encode_base64(self, embeddings: numpy.ndarray) -> List[str]:
        return [
            base64.b64encode(e.tobytes()).decode("utf-8") for e in embeddings
        ]

def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = await asyncio.to_thread(worker.get_embeddings, params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)

@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = BERTWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
