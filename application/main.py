import os
import io
import logging
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse
from application.inference import InferenceOnSingleImage

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

endpoint_prefix = os.environ.get("ENDPOINT_PREFIX", "/image-captioning")

app = FastAPI(docs_url=f"{endpoint_prefix}",
              title="Image Captioning Application",
              description="This is a simple FastAPI application for image captioning",
              openapi_url=f"{endpoint_prefix}/openapi.json"
              )

origins = [
    "http://localhost:8081/image-captioning",
    "http://0.0.0.0:8081/image-captioning"
    "http://localhost:8081",
    "http://0.0.0.0:8081"
    "http://127.0.0.1:8081",
    "http://127.0.0.1:8081/image-captioning",


]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=3600,
)
templates = Jinja2Templates(directory="templates")


@app.post(f"{endpoint_prefix}/caption_image/", tags=["Image Captioning"])  # response_class=HTMLResponse)
async def image_file_text(image_file: UploadFile = File(...)):
    logging.info(image_file.file)
    try:
        os.mkdir("./resources/image_file")

        logging.info(os.getcwd())
    except Exception as e:
        logging.info(e)

    file_path = os.path.join(os.getcwd(), "./resources/image_file", image_file.filename.replace(" ", "-"))
    with open(file_path, 'wb+') as f:
        f.write(image_file.file.read())

    image = Image.open(os.path.join(file_path)).convert('RGB')
    get_caption = InferenceOnSingleImage()
    orig_image, sentence = get_caption.caption_sentence_from_upload(image)

    output = {
        "filename": image_file.filename,
        "predicted_caption": sentence
    }

    logging.info(f"Prediction for {image_file} ...done!")
    return output
    # return templates.TemplateResponse("page.html", {"request": image_file.filename, "data": sentence})


@app.post(f"{endpoint_prefix}/upload_image/", tags=["Image Captioning"])
async def image_file_preview(image_file: UploadFile = File(...)):
    logging.info(image_file.file)
    try:
        os.mkdir("./resources/image_file")

        logging.info(os.getcwd())
    except Exception as e:
        logging.info(e)

    file_path = os.path.join(os.getcwd(), "./resources/image_file", image_file.filename.replace(" ", "-"))
    with open(file_path, 'wb+') as f:
        f.write(image_file.file.read())

    image = Image.open(os.path.join(file_path)).convert('RGB')
    get_caption = InferenceOnSingleImage()
    orig_image, sentence = get_caption.caption_sentence_from_upload(image)

    output = {
        "filename": image_file.filename,
        "predicted_caption": sentence
    }

    image.thumbnail((200, 200))
    buf = io.BytesIO()
    image.save(buf, "JPEG")
    buf.seek(0)
    logging.info(f"Prediction for {image_file} ...done!")
    return StreamingResponse(content=buf, media_type="image/jpeg", headers={"Content-Disposition": f"{output}"})


@app.post(f"{endpoint_prefix}/upload_image/metadata", tags=["Image Captioning"])
async def image_metadata(image_file: UploadFile = File(...)):
    """Endpoint to get image metadata and caption"""
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        get_caption = InferenceOnSingleImage()
        orig_image, sentence = get_caption.caption_sentence_from_upload(image)

        return JSONResponse(content={
            "filename": image_file.filename,
            "predicted_caption": sentence
        })
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{endpoint_prefix}/upload_image/thumbnail", tags=["Image Captioning"])
async def image_thumbnail(image_file: UploadFile = File(...)):
    """Endpoint to get image thumbnail"""
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Create thumbnail
        image.thumbnail((200, 200))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(
            content=buf,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="thumbnail-{image_file.filename}"',
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
