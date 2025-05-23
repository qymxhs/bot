# FastAPI + OpenAI 后端实现
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import uvicorn
from dotenv import load_dotenv
import logging
import uuid
import aiofiles

# 加载环境变量
load_dotenv()

# 配置OpenAI API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")

# 创建OpenAI客户端
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="ChatGPT API")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置CORS允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建临时文件目录
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 模型定义
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# 验证API密钥中间件
async def verify_api_key():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API密钥未配置")
    return OPENAI_API_KEY

@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """处理纯文本聊天请求"""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 可调整为适合的模型
            messages=messages
        )
        
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"聊天请求错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理聊天请求时出错: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """处理单个文件上传"""
    try:
        # 检查文件大小
        contents = await file.read()
        file_size = len(contents)
        if file_size > 10 * 1024 * 1024:  # 10MB 限制
            raise HTTPException(status_code=413, detail="文件大小超过限制（最大10MB）")
                
        # 生成唯一文件名避免冲突
        file_id = uuid.uuid4()
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        # 保存上传的文件
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            await out_file.write(contents)
            
        # 重置文件指针，以便后续读取
        await file.seek(0)
        
        # 根据文件类型处理
        if file.content_type.startswith("text/"):
            # 处理文本文件
            async with aiofiles.open(temp_file_path, 'r', encoding='utf-8') as text_file:
                file_content = await text_file.read()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "分析用户上传的文本文件并提供见解。"},
                    {"role": "user", "content": f"以下是文件 {file.filename} 的内容:\n\n{file_content}"}
                ]
            )
            
        elif file.content_type.startswith("image/"):
            # 处理图片文件
            with open(temp_file_path, "rb") as image_file:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"描述这个文件: {file.filename}"},
                                {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{content.decode('latin1')}"}}
                            ]
                        }
                    ]
                )
        else:
            # 其他类型文件的处理
            return JSONResponse(
                status_code=200,
                content={"response": f"已接收文件 {file.filename}，但该文件类型暂不支持直接分析。"}
            )
        
        # 清理临时文件
        os.remove(temp_file_path)
        
        return {
            "response": response.choices[0].message.content, 
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"文件上传错误: {str(e)}")
        # 确保清理任何临时文件
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {str(e)}")

@app.post("/chat-with-files")
async def chat_with_files(
    message: str = Form(...),
    files: List[UploadFile] = File([]),
    api_key: str = Depends(verify_api_key)
):
    """处理包含文件的聊天请求"""
    temp_files = []
    
    try:
        # 构建消息内容
        content = [{"type": "text", "text": message}]
        
        # 处理上传的文件
        for file in files:
            file_id = uuid.uuid4()
            file_extension = os.path.splitext(file.filename)[1]
            temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
            temp_files.append(temp_path)
            
            # 保存文件
            async with aiofiles.open(temp_path, 'wb') as out_file:
                file_content = await file.read()
                await out_file.write(file_content)
            
            if file.content_type.startswith("image/"):
                # 添加图像到内容
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{file.content_type};base64,{file_content.decode('latin1')}"}
                })
            elif file.content_type.startswith("text/"):
                # 添加文本文件内容
                async with aiofiles.open(temp_path, 'r', encoding='utf-8') as text_file:
                    text_content = await text_file.read()
                content.append({"type": "text", "text": f"文件 {file.filename} 内容:\n{text_content}"})
        
        # 确定使用的模型
        model = "gpt-4o" if any(file.content_type.startswith("image/") for file in files) else "gpt-4o-mini"
        
        # 调用OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个可以分析文本和图像的AI助手。"},
                {"role": "user", "content": content}
            ]
        )
        
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"处理带文件的聊天请求错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")
    finally:
        # 清理所有临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# 在所有路由之前挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """重定向到首页"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
