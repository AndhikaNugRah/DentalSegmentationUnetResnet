from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()

# CORS settings
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model
onnx_model_path = "C:\Users\abc\Downloads\best_checkpoint.ckpt"
session = ort.InferenceSession(onnx_model_path)


app.mount("/static", StaticFiles(directory="static"), name="static")


# Preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((512, 256))  # Resize to the input size of the model
    image = np.array(image).astype('float32') / 255.0  # Normalize the image
    image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Postprocessing function
def postprocess_output(output):
    output = np.squeeze(output, axis=0)  # Remove batch dimension
    output = np.argmax(output, axis=0)  # Get the class with highest probability
    return output


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(io.BytesIO(await file.read()))

        # Preprocess the image
        input_image = preprocess_image(image)

        # Run the model
        ort_inputs = {session.get_inputs()[0].name: input_image}
        ort_outs = session.run(None, ort_inputs)

        # Post-process the output
        prediction = np.squeeze(ort_outs[0])  # Remove batch dimension
        prediction = (prediction > 0.5).astype(np.uint8) * 255  # Binarize and scale to 255

        # Convert the prediction to an image
        pred_image = Image.fromarray(prediction)

        # Save the image to a bytes buffer
        buffer = io.BytesIO()
        pred_image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())
