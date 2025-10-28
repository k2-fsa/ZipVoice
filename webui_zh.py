# webui_zh
# Created by @ByronLeeeee

import gradio as gr
import subprocess
import os
import time
import torch
import sys
import logging
import pandas as pd
import shutil
import zipfile

# --- 全局设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 确保输出目录存在
os.makedirs("outputs", exist_ok=True)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 设备信息检查 ---
def get_device_info():
    """检查并返回当前 Torch 使用的设备信息。"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        pytorch_cuda_version = torch.version.cuda
        return f"""
        <div>
        ✅ CUDA 可用! 正在使用 GPU 加速。<br>
        GPU: {gpu_name} | PyTorch CUDA: {pytorch_cuda_version}
        </div>
        """
    else:
        return """
        <div>
        ⚠️ CUDA 不可用! 程序将运行在 CPU 上，建议使用ONNX模式进行推理。
        </div>
        """

# --- 核心命令行执行函数 ---
def run_command(command, progress_desc="正在合成..."):
    """执行命令行命令并记录输出。"""
    logging.info(f"执行命令: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"命令执行失败! 返回码: {process.returncode}")
            logging.error(f"标准输出:\n{stdout}")
            logging.error(f"标准错误:\n{stderr}")
            raise gr.Error(f"后端脚本执行失败: {stderr[:1000]}")
        else:
            logging.info(f"脚本标准输出:\n{stdout}")
            if stderr:
                logging.warning(f"脚本标准错误输出:\n{stderr}")
                
    except FileNotFoundError:
        logging.error(f"命令未找到: {command[0]}")
        raise gr.Error(f"无法执行命令。请确保您在正确的 ZipVoice 环境和目录中运行此脚本。")
    except Exception as e:
        logging.error(f"执行命令时发生未知错误: {e}")
        raise gr.Error(f"发生未知错误: {str(e)}")


# --- 各个标签页的后端函数 ---

def inference_single_speaker_cli(
    model_name, prompt_audio_path, prompt_text, target_text,
    guidance_scale, num_step, speed, progress=gr.Progress()
):
    if not all([prompt_audio_path, prompt_text, target_text]):
        raise gr.Error("请提供所有输入：参考音频、参考音频文本和目标文本。")
    progress(0.1, desc="准备参数...")
    output_filename = f"outputs/output_single_{int(time.time())}.wav"
    python_executable = sys.executable
    command = [
        python_executable, "-m", "zipvoice.bin.infer_zipvoice",
        "--model-name", model_name,
        "--prompt-wav", prompt_audio_path,
        "--prompt-text", prompt_text,
        "--text", target_text,
        "--res-wav-path", output_filename,
        "--guidance-scale", str(guidance_scale),
        "--num-step", str(int(num_step)),
        "--speed", str(speed)
    ]
    progress(0.5, desc="正在合成...")
    run_command(command)
    progress(1.0, desc="完成！")
    return output_filename

def inference_onnx_cli(
    model_name, use_int8, prompt_audio_path, prompt_text, target_text,
    guidance_scale, num_step, speed, progress=gr.Progress()
):
    if not all([prompt_audio_path, prompt_text, target_text]):
        raise gr.Error("请提供所有输入：参考音频、参考音频文本和目标文本。")
    progress(0.1, desc="准备参数...")
    output_filename = f"outputs/output_onnx_{int(time.time())}.wav"
    python_executable = sys.executable
    command = [
        python_executable, "-m", "zipvoice.bin.infer_zipvoice_onnx",
        "--model-name", model_name,
        "--onnx-int8", str(use_int8),
        "--prompt-wav", prompt_audio_path,
        "--prompt-text", prompt_text,
        "--text", target_text,
        "--res-wav-path", output_filename,
        "--guidance-scale", str(guidance_scale),
        "--num-step", str(int(num_step)),
        "--speed", str(speed)
    ]
    progress(0.5, desc="正在合成 (ONNX)...")
    run_command(command)
    progress(1.0, desc="完成！")
    return output_filename

def inference_dialogue_cli(
    model_name, prompt_type,
    merged_prompt_audio_path, merged_prompt_text,
    spk1_prompt_audio_path, spk1_prompt_text,
    spk2_prompt_audio_path, spk2_prompt_text,
    dialogue_text,
    guidance_scale, num_step, speed, progress=gr.Progress()
):
    if not dialogue_text:
        raise gr.Error("请输入要合成的对话文本。")
    progress(0.1, desc="创建临时任务文件...")
    
    temp_tsv_filename = f"temp_dialog_list_{int(time.time())}.tsv"
    output_wav_name = f"dialogue_{int(time.time())}"
    output_dir = "outputs"
    output_filename = os.path.join(output_dir, f"{output_wav_name}.wav")

    try:
        with open(temp_tsv_filename, "w", encoding="utf-8") as f:
            if prompt_type == "合并的Prompt":
                if not all([merged_prompt_audio_path, merged_prompt_text]):
                    raise gr.Error("请提供合并的参考音频和文本。")
                line = f"{output_wav_name}\t{merged_prompt_text}\t{merged_prompt_audio_path}\t{dialogue_text}"
                f.write(line)
            else:
                if not all([spk1_prompt_audio_path, spk1_prompt_text, spk2_prompt_audio_path, spk2_prompt_text]):
                    raise gr.Error("请为两位说话人提供完整的参考音频和文本。")
                line = f"{output_wav_name}\t{spk1_prompt_text}\t{spk2_prompt_text}\t{spk1_prompt_audio_path}\t{spk2_prompt_audio_path}\t{dialogue_text}"
                f.write(line)

        python_executable = sys.executable
        command = [
            python_executable, "-m", "zipvoice.bin.infer_zipvoice_dialog",
            "--model-name", model_name,
            "--test-list", temp_tsv_filename,
            "--res-dir", output_dir,
            "--guidance-scale", str(guidance_scale),
            "--num-step", str(int(num_step)),
            "--speed", str(speed)
        ]
        
        progress(0.5, desc="正在合成对话...")
        run_command(command)
        
    finally:
        if os.path.exists(temp_tsv_filename):
            os.remove(temp_tsv_filename)

    progress(1.0, desc="完成！")
    return output_filename

def inference_batch_cli(
    task_type, model_name, tsv_file, dataframe, progress=gr.Progress()
):
    if tsv_file is None and (dataframe is None or dataframe.empty):
        raise gr.Error("请上传一个 TSV 文件或在编辑器中创建数据。")

    progress(0.1, desc="准备批量任务...")
    
    temp_tsv_filename = f"temp_batch_list_{int(time.time())}.tsv"
    
    if tsv_file is not None:
        # 如果上传了文件，使用它
        shutil.copy(tsv_file.name, temp_tsv_filename)
    else:
        # 否则，使用 DataFrame 的内容
        dataframe.to_csv(temp_tsv_filename, sep='\t', header=False, index=False)
        
    batch_id = f"batch_{int(time.time())}"
    output_dir = os.path.join("outputs", batch_id)
    os.makedirs(output_dir, exist_ok=True)
    
    python_executable = sys.executable
    
    if task_type == "单人语音":
        script_name = "zipvoice.bin.infer_zipvoice"
    else: # 对话
        script_name = "zipvoice.bin.infer_zipvoice_dialog"

    command = [
        python_executable, "-m", script_name,
        "--model-name", model_name,
        "--test-list", temp_tsv_filename,
        "--res-dir", output_dir
    ]

    try:
        total_lines = sum(1 for line in open(temp_tsv_filename, 'r', encoding='utf-8'))
        progress(0.5, desc=f"正在处理 {total_lines} 条音频...")
        run_command(command, progress_desc=f"正在处理 {total_lines} 条音频...")
        
        progress(0.9, desc="正在打包结果...")
        zip_path = os.path.join("outputs", f"{batch_id}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)
        
    finally:
        if os.path.exists(temp_tsv_filename):
            os.remove(temp_tsv_filename)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir) # 删除临时文件夹

    progress(1.0, desc="批量处理完成！")
    return gr.update(value=zip_path, visible=True)

# --- UI 辅助函数 ---
def update_single_speaker_defaults(model_name):
    """根据单人 TTS 模型名称更新高级参数的默认值。"""
    if model_name == "zipvoice":
        return gr.update(value=1.0), gr.update(value=16)
    elif model_name == "zipvoice_distill":
        return gr.update(value=3.0), gr.update(value=8)
    # 默认回退
    return gr.update(), gr.update()

# --- Gradio UI 界面定义 ---
with gr.Blocks(theme=gr.themes.Soft(), title="ZipVoice WebUI") as app:
    gr.Markdown("# ⚡ ZipVoice 语音合成 WebUI")
    gr.Markdown("这是一个基于 [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) 项目的 WebUI。")
    gr.Markdown(value=get_device_info())

    with gr.Tabs():
        # --- 单人语音合成 (PyTorch) ---
        with gr.TabItem("1. 单人语音合成 (PyTorch)"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 输入")
                    model_name_single = gr.Dropdown(["zipvoice_distill", "zipvoice"], value="zipvoice_distill", label="模型")
                    prompt_audio_single = gr.Audio(label="参考音频", type="filepath", sources=["upload", "microphone"])
                    prompt_text_single = gr.Textbox(label="参考音频的文本", placeholder="输入参考音频对应的文本...")
                    target_text_single = gr.Textbox(label="目标文本", placeholder="输入想要合成的文本...", lines=3)
                    with gr.Accordion("高级设置", open=False):
                        guidance_scale_single = gr.Slider(minimum=0.5, maximum=5.0, value=3.0, step=0.1, label="引导系数")
                        num_step_single = gr.Slider(minimum=2, maximum=20, value=8, step=1, label="步数")
                        speed_single = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="语速")
                    submit_btn_single = gr.Button("合成语音", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 输出")
                    output_audio_single = gr.Audio(label="生成的音频")

        # --- 单人语音合成 (ONNX) ---
        with gr.TabItem("2. 单人语音合成 (ONNX CPU)"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 输入 (ONNX)")
                    model_name_onnx = gr.Dropdown(["zipvoice_distill", "zipvoice"], value="zipvoice_distill", label="模型")
                    use_int8_onnx = gr.Checkbox(label="使用 INT8 量化模型 (更快)", value=False)
                    prompt_audio_onnx = gr.Audio(label="参考音频", type="filepath", sources=["upload", "microphone"])
                    prompt_text_onnx = gr.Textbox(label="参考音频的文本", placeholder="输入参考音频对应的文本...")
                    target_text_onnx = gr.Textbox(label="目标文本", placeholder="输入想要合成的文本...", lines=3)
                    with gr.Accordion("高级设置", open=False):
                        guidance_scale_onnx = gr.Slider(minimum=0.5, maximum=5.0, value=3.0, step=0.1, label="引导系数")
                        num_step_onnx = gr.Slider(minimum=2, maximum=20, value=8, step=1, label="步数")
                        speed_onnx = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="语速")
                    submit_btn_onnx = gr.Button("合成语音 (ONNX)", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 输出")
                    output_audio_onnx = gr.Audio(label="生成的音频")

        # --- 对话语音合成 ---
        with gr.TabItem("3. 对话语音合成"):
             with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 输入")
                    model_name_dialogue = gr.Dropdown(["zipvoice_dialog", "zipvoice_dialog_stereo"], value="zipvoice_dialog", label="模型")
                    prompt_type = gr.Radio(["合并的Prompt", "分离的Prompt"], label="参考音频类型", value="分离的Prompt")
                    with gr.Group(visible=False) as merged_prompt_group:
                        merged_prompt_audio = gr.Audio(label="参考音频 (合并)", type="filepath", sources=["upload"])
                        merged_prompt_text = gr.Textbox(label="参考音频文本 (合并)", placeholder="例如: [S1] 你好。[S2] 你好呀。")
                    with gr.Group(visible=True) as splitted_prompt_group:
                        with gr.Row():
                            with gr.Column():
                                spk1_prompt_audio = gr.Audio(label="说话人1 参考音频", type="filepath", sources=["upload"])
                                spk1_prompt_text = gr.Textbox(label="说话人1 参考文本")
                            with gr.Column():
                                spk2_prompt_audio = gr.Audio(label="说话人2 参考音频", type="filepath", sources=["upload"])
                                spk2_prompt_text = gr.Textbox(label="说话人2 参考文本")
                    dialogue_text = gr.Textbox(label="要合成的对话文本", placeholder="例如: [S1] 我很好，你呢？[S2] 我也很好。", lines=4)
                    with gr.Accordion("高级设置", open=False):
                        guidance_scale_dialogue = gr.Slider(minimum=0.5, maximum=5.0, value=1.5, step=0.1, label="引导系数")
                        num_step_dialogue = gr.Slider(minimum=2, maximum=20, value=16, step=1, label="步数")
                        speed_dialogue = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="语速")
                    submit_btn_dialogue = gr.Button("生成对话", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 输出")
                    output_audio_dialogue = gr.Audio(label="生成的对话音频")

        # --- 批量推理 ---
        with gr.TabItem("4. 批量推理 (TSV)"):
            gr.Markdown("在此处创建或上传TSV文件进行批量推理。处理完成后，结果将打包为 ZIP 文件供下载。")
            with gr.Row():
                with gr.Column(scale=2):
                    batch_task_type = gr.Radio(["单人语音", "对话"], label="任务类型", value="单人语音")
                    batch_model_name = gr.Dropdown(
                        ["zipvoice_distill", "zipvoice", "zipvoice_dialog", "zipvoice_dialog_stereo"],
                        value="zipvoice_distill",
                        label="模型"
                    )
                    
                    gr.Markdown("#### 编辑或上传您的 TSV 文件")
                    upload_btn = gr.UploadButton("上传 TSV 文件", file_types=[".tsv"])
                    
                    df_single_headers = ["wav_name", "prompt_transcription", "prompt_wav", "text"]
                    df_dialogue_headers = ["wav_name", "spk1_prompt_transcription", "spk2_prompt_transcription", "spk1_prompt_wav", "spk2_prompt_wav", "text"]
                    
                    dataframe_editor = gr.DataFrame(
                        headers=df_single_headers,
                        datatype=["str"] * len(df_single_headers),
                        row_count=(2, "dynamic"),
                        col_count=(len(df_single_headers), "fixed"),
                        label="TSV 编辑器",
                        wrap=True
                    )
                    
                    submit_btn_batch = gr.Button("开始批量推理", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 输出")
                    batch_output_file = gr.File(label="下载结果 (ZIP)", visible=False)
                    gr.Markdown(
                        """
                        **TSV 格式说明:**
                        
                        **单人语音:**
                        `输出文件名\t参考文本\t参考音频路径\t目标文本`
                        
                        **对话 (分离Prompt):**
                        `输出文件名\t说话人1文本\t说话人2文本\t说话人1音频路径\t说话人2音频路径\t对话文本`
                        
                        *注意: 音频路径应为本地绝对路径或相对于此脚本的相对路径。*
                        """
                    )
                    
        # --- 使用说明 ---
        with gr.TabItem("5. 使用说明"):
            gr.Markdown("""
            ## 如何使用 ZipVoice WebUI

            ### 1. 单人语音合成
            - **功能**: 克隆一个人的声音并用其朗读新的文本。
            - **步骤**:
              1. 在 **"1. 单人语音合成"** 标签页中，选择一个模型 (`zipvoice_distill` 速度更快)。
              2. 上传一段 **参考音频** (Prompt Audio)，时长建议在 3-10 秒。
              3. 在 **参考音频的文本** 框中，准确输入该参考音频对应的文字。
              4. 在 **目标文本** 框中，输入您希望模型朗读的新内容。
              5. 点击 **合成语音**。

            ### 2. 对话语音合成
            - **功能**: 生成包含两个不同说话人的对话。
            - **步骤**:
              1. 在 **"3. 对话语音合成"** 标签页中，选择模型 (`zipvoice_dialog_stereo` 会生成双声道音频)。
              2. 选择 **参考音频类型**：
                 - **分离的Prompt (推荐)**: 分别上传两位说话人的独立音频和对应文本。
                 - **合并的Prompt**: 上传一个包含两人对话的音频，并在文本框中用 `[S1]` 和 `[S2]` 标注。
              3. 在 **要合成的对话文本** 框中，输入完整的对话内容，并使用 `[S1]` 和 `[S2]` 来区分不同说话人的轮次。
                 - **示例**: `[S1] 你好吗？[S2] 我很好，谢谢。[S1] 不客气。`
              4. 点击 **生成对话**。

            ### 3. ONNX CPU 推理
            - **功能**: 与单人语音合成相同，但使用 ONNX 模型在 CPU 上运行，通常比 PyTorch 在 CPU 上的推理速度更快。
            - **步骤**:
              1. 在 **"2. 单人语音合成 (ONNX CPU)"** 标签页中操作。
              2. 勾选 **使用 INT8 量化模型** 可以获得更快的速度，但可能会牺牲一点点质量。
              3. 其余步骤与单人语音合成完全相同。
            
            ### 4. 批量推理
            - **功能**: 一次性处理多个合成任务。
            - **步骤**:
              1. 在 **"4. 批量推理 (TSV)"** 标签页中，首先选择 **任务类型** (单人或对话) 和模型。
              2. **创建数据**: 在 **TSV 编辑器** 中按照格式说明手动输入多行任务。
              3. **或上传数据**: 点击 **上传 TSV 文件** 按钮，选择一个本地的制表符分隔文件。
              4. 点击 **开始批量推理**。任务完成后，右侧会提供一个包含所有生成音频的 ZIP 文件供下载。

            ### 纠正中文多音字发音
            当遇到中文多音字发音错误时，您可以通过 pinyin 手动指定。
            - **格式**: `这把剑<chang2>三十公分`
            - **说明**: 用尖括号 `< >` 包围正确的拼音，并在末尾加上声调数字 (1-4 为四声，5 为轻声)。
            """)

    # --- 事件处理逻辑 ---
    submit_btn_single.click(
        fn=inference_single_speaker_cli,
        inputs=[model_name_single, prompt_audio_single, prompt_text_single, target_text_single, guidance_scale_single, num_step_single, speed_single],
        outputs=output_audio_single
    )
    submit_btn_onnx.click(
        fn=inference_onnx_cli,
        inputs=[model_name_onnx, use_int8_onnx, prompt_audio_onnx, prompt_text_onnx, target_text_onnx, guidance_scale_onnx, num_step_onnx, speed_onnx],
        outputs=output_audio_onnx
    )
    submit_btn_dialogue.click(
        fn=inference_dialogue_cli,
        inputs=[model_name_dialogue, prompt_type, merged_prompt_audio, merged_prompt_text, spk1_prompt_audio, spk1_prompt_text, spk2_prompt_audio, spk2_prompt_text, dialogue_text, guidance_scale_dialogue, num_step_dialogue, speed_dialogue],
        outputs=output_audio_dialogue
    )

    model_name_single.change(
        fn=update_single_speaker_defaults,
        inputs=model_name_single,
        outputs=[guidance_scale_single, num_step_single]
    )
    model_name_onnx.change(
        fn=update_single_speaker_defaults,
        inputs=model_name_onnx,
        outputs=[guidance_scale_onnx, num_step_onnx]
    )
    
    def toggle_prompt_type(choice):
        return gr.update(visible=choice == "合并的Prompt"), gr.update(visible=choice == "分离的Prompt")

    prompt_type.change(fn=toggle_prompt_type, inputs=prompt_type, outputs=[merged_prompt_group, splitted_prompt_group])
    
    def update_dataframe(task_type):
        if task_type == "单人语音":
            headers = df_single_headers
        else: # 对话
            headers = df_dialogue_headers
        return gr.update(headers=headers, col_count=(len(headers), "fixed"), value=None)

    batch_task_type.change(fn=update_dataframe, inputs=batch_task_type, outputs=dataframe_editor)
    
    def upload_file_to_df(file, task_type):
        if file is None:
            return None
        try:
            df = pd.read_csv(file.name, sep='\t', header=None)
            if task_type == "单人语音":
                expected_cols = 4
            else: # 对话
                expected_cols = 6
            
            if df.shape[1] != expected_cols:
                raise gr.Error(f"TSV 文件列数错误！'{task_type}' 任务需要 {expected_cols} 列，但文件有 {df.shape[1]} 列。")
            
            return gr.update(value=df)
        except Exception as e:
            raise gr.Error(f"读取或解析 TSV 文件失败: {e}")

    upload_btn.upload(fn=upload_file_to_df, inputs=[upload_btn, batch_task_type], outputs=dataframe_editor)
    
    submit_btn_batch.click(
        fn=inference_batch_cli,
        inputs=[batch_task_type, batch_model_name, upload_btn, dataframe_editor],
        outputs=batch_output_file
    )

if __name__ == "__main__":
    app.launch()