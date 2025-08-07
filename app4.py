import streamlit as st
import pandas as pd
import json
from modules.data_loader import DataLoader
from modules.preprocessor import DataPreprocessor
from modules.analyzer import DataAnalyzer
from modules.visualizer import DataVisualizer
from modules.nlp_interface import NLPInterface
from modules.db_manager import DBManager
from modules.ui_components import UIComponents  # 导入UI组件
from modules.model4 import RegressionModel, TimeSeriesModel
import streamlit.components.v1 as components  # 添加组件导入
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
import logging
import io
import base64
import time
import requests

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score)

import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tsa.stattools import adfuller, acf, q_stat
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
from plotly.subplots import make_subplots

model_functions ={"对话":["azureopenai","qwen","deepseek","zhipu","doubao","localmodel"],
                  "文件":["qwen"],
                  "图片":["qwen","doubao","localmodel"],
                  "联网":["qwen"],
                  "默认":["azureopenai","qwen","deepseek","zhipu","doubao","localmodel"]}

# 模拟一个简单的 NLP API 调用函数
def call_nlp_api(api_url, api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        print(f"API 调用失败: {e}")
        return False

# 添加一个辅助函数来初始化数据相关组件
def initialize_data_components(data):
    # 更新数据和相关组件
    st.session_state.data = data
    st.session_state.preprocessor = DataPreprocessor(data)
    st.session_state.analyzer = DataAnalyzer(data)
    st.session_state.visualizer = DataVisualizer(data)
    
    # 更新NLP接口
    st.session_state.nlp_interface.set_data(data)
    st.session_state.nlp_interface.set_preprocessor(st.session_state.preprocessor)
    st.session_state.nlp_interface.set_analyzer(st.session_state.analyzer)
    st.session_state.nlp_interface.set_visualizer(st.session_state.visualizer)

# 添加一个辅助函数来更新用户数据和相关组件
def update_session_data(new_data):
    initialize_data_components(new_data)
    # 刷新页面
    # st.rerun()

# 处理用户查询的函数
def process_user_query(user_query):
    try:
        # 首先进行意图识别
        result = st.session_state.nlp_interface.process_query(user_query)
        
        # 检查是否有错误
        if 'error' in result:
            # 如果是意图识别错误，尝试与智谱AI直接对话
            if result.get('error') == '未能识别意图' or '未知的' in result.get('message', ''):
                try:
                    # 使用chat_directly方法直接与智谱AI对话
                    # 检查是否已有直接响应
                    if 'response_directly' in result and result['response_directly']:
                        response = "您的问题已在右侧显示"
                    else:
                        direct_response = st.session_state.nlp_interface.chat_directly(
                            query=user_query,
                            system_prompt="你是一个专业的数据分析助手。请简洁、准确地回答用户关于数据分析的问题。如果用户询问的功能超出系统能力，请尝试提供有用的信息。"
                        )
                        # 将直接回答放入result
                        result['response_directly'] = direct_response
                        response = "您的问题已在右侧显示"
                        
                    # 确保结果显示在右侧
                    st.session_state.ai_result = result
                except Exception as e:
                    response = f"智谱AI对话失败: {str(e)}"
            else:
                response = f"处理请求时出错: {result['message']}"
                st.error(f"错误详情: {result['error']}")
        else:
            response = result['message']
            # 如果结果中包含图表或数据，在主区域显示
            if any(key in result for key in [
                'figure', 'stats', 'correlation', 'results', 'summary', 'response_directly',
                'show_missing_values_ui', 'show_duplicates_ui', 'show_feature_conversion_ui',
                'show_descriptive_stats_ui', 'show_correlation_ui', 'show_statistical_tests_ui',
                'show_outliers_ui', 'charts', 'html'
            ]):
                st.session_state.ai_result = result
                # 添加一个引导提示，指导用户查看主区域的结果
                if 'stats' in result:
                    response += "\n\n结果已在页面右侧显示，请查看"
                else:
                    response += "\n\n结果已在主区域显示"
        
        return response, result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"错误详情: {str(e)}\n\n{error_details}")
        return f"处理请求时出现错误，请查看错误详情", None

# 重置NLP接口状态的函数
def reset_nlp_interface():
    """重置NLP接口的各种状态设置为默认值"""
    st.session_state.nlp_interface.local_llm_file_path = ""  # 清空文件路径
    st.session_state.nlp_interface.local_llm_image_path = ""  # 清空图片路径
    st.session_state.nlp_interface.enable_llm_internet = False  # 禁用搜索引擎
    st.session_state.nlp_interface.chat_llm_directly = False  # 清除直接对话模式
    st.session_state.nlp_interface.clear_llm_history = False  # 默认不清除历史

# 显示特定类型的UI组件函数
def show_ui_component(component_type, result=None):
    """
    显示特定类型的UI组件并处理可能的数据更新
    
    参数:
    component_type (str): UI组件类型，如'missing_values', 'duplicates', 'feature_conversion', 'outliers'
    result (dict): 包含相关数据的结果字典
    
    返回:
    None
    """
    new_data = None
    
    if component_type == 'missing_values':
        # 如果有缺失值列表，先显示统计信息
        if result and 'missing_columns' in result and len(result['missing_columns']) > 0 and 'filled_count' not in result:
            st.subheader("存在缺失值的列")
            missing_info = []
            for col in result['missing_columns']:
                if isinstance(result['stats'], list):
                    for stat in result['stats']:
                        if stat['列名'] == col:
                            missing_info.append({
                                '列名': col,
                                '缺失值数量': stat['缺失值数量'],
                                '缺失值比例(%)': stat['缺失值比例(%)']
                            })
                            break
            
            if missing_info:
                st.dataframe(pd.DataFrame(missing_info))
        
        # 显示缺失值处理UI
        new_data = UIComponents.show_missing_values_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="ui_")
    
    elif component_type == 'duplicates':
        # 显示重复值处理UI
        new_data = UIComponents.show_duplicates_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="ui_")
    
    elif component_type == 'feature_conversion':
        # 显示特征类型转换UI
        new_data = UIComponents.show_feature_conversion_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="ui_")
    
    elif component_type == 'outliers':
        # 显示异常值处理UI
        new_data = UIComponents.show_outliers_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="ui_")
    
    elif component_type == 'descriptive_stats':
        # 显示描述性统计UI
        UIComponents.show_descriptive_stats_ui(st.session_state.analyzer, st.session_state.data)
    
    elif component_type == 'correlation':
        # 显示相关性分析UI
        UIComponents.show_correlation_ui(st.session_state.analyzer, st.session_state.data)
    
    elif component_type == 'statistical_test':
        # 显示统计检验UI
        UIComponents.show_statistical_tests_ui(st.session_state.analyzer, st.session_state.data)
    
    # 如果有新数据且不为None，则更新session状态中的数据
    if new_data is not None:
        update_session_data(new_data)

# 设置页面配置
st.set_page_config(
    page_title="人工智能数据分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置页面样式
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'nlp_interface' not in st.session_state:
    # 更新API密钥格式，使用新版API
    # 注意：这是一个示例API密钥，为了演示目的使用
    # 实际使用时请配置有效的API密钥
    # api_key = st.secrets["zhipuai_api_key"] if "zhipuai_api_key" in st.secrets else "请在此处填入有效的API密钥"
    api_key = "205218f94d11b30a98b9c99e9c42e845.T5tKDlsCpvCjAUZc"
    st.session_state.nlp_interface = NLPInterface(api_key)
    # 默认使用智谱模型
    # st.session_state.nlp_interface.model = "azureopenai"
    st.session_state.nlp_interface.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    st.session_state.nlp_interface.api_key = "205218f94d11b30a98b9c99e9c42e845.T5tKDlsCpvCjAUZc"

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'db_manager' not in st.session_state:
    # 初始化数据库管理器，默认自动选择合适的数据库类型
    try:
        st.session_state.db_manager = DBManager(
            host="localhost",
            user="root",
            password="password",
            database="data_analysis",
            db_type="auto"  # 自动选择MySQL或SQLite
        )
    except Exception as e:
        st.session_state.db_manager = None
        logger.error(f"数据库初始化失败: {e}")
if 'db_tables' not in st.session_state:
    st.session_state.db_tables = []
if 'current_db_table' not in st.session_state:
    st.session_state.current_db_table = None
if 'sql_history' not in st.session_state:
    st.session_state.sql_history = []

# 页面标题
st.title("📊 智能数据分析师")

# 侧边栏
with st.sidebar:
    st.header("功能导航")
    page = st.radio(
        "选择功能",
        ["数据加载", "数据预处理", "数据分析", "数据可视化", "数据建模",  "SQL数据库"]
    )    
    
    # 在侧边栏底部添加智能助手对话框
    st.markdown("---")
    st.header("💬 智能助手")
    
    # 初始化对模型选择的追踪
    if 'previous_model_selection' not in st.session_state:
        st.session_state.previous_model_selection = None
        st.session_state.is_first_model_load = True
    
    # 直接添加模型选择下拉框
    current_model = st.session_state.nlp_interface.model if hasattr(st.session_state.nlp_interface, 'model') else "azureopenai"

    new_model = st.selectbox(
        "选择模型",
        ["OpenAI", "通义", "DeepSeek", "智谱", "豆包", "本地模型"],
        index=["azureopenai", "qwen", "deepseek", "zhipu", "doubao", "localmodel"].index(current_model) if current_model in ["azureopenai", "qwen", "deepseek", "zhipu", "doubao", "localmodel"] else 0
    )

    # 将显示值映射到代码值
    model_mapping = {
        "OpenAI": "azureopenai",
        "通义": "qwen",
        "DeepSeek": "deepseek",
        "智谱": "zhipu",
        "豆包": "doubao",
        "本地模型": "localmodel"
    }
    
    # 检测模型选择变化并自动更新
    if new_model != st.session_state.previous_model_selection and not st.session_state.is_first_model_load:
        # 更新上一次选择的值
        st.session_state.previous_model_selection = new_model
        
        # 更新模型并重新初始化
        model_code = model_mapping[new_model]
        st.session_state.nlp_interface.model = model_code
        st.session_state.nlp_interface._initialize_api_client()
        print(f"{st.session_state.nlp_interface.api_url}, {st.session_state.nlp_interface.api_key}")
        print(f"{st.session_state.nlp_interface.client}, {st.session_state.nlp_interface.api_key}")
        print("st.session_state.nlp_interface.api_working", st.session_state.nlp_interface.api_working)
        st.session_state.nlp_interface.api_working = call_nlp_api(st.session_state.nlp_interface.api_url, st.session_state.nlp_interface.api_key)
        st.success(f"已切换到{new_model}模型" + (" 并成功验证" if st.session_state.nlp_interface.api_working else "，但验证失败"))
        # 重新加载页面以应用更改
        # st.rerun()
    elif st.session_state.is_first_model_load:
        # 首次加载时，仅记录当前选择，不执行更新
        st.session_state.is_first_model_load = False
        st.session_state.previous_model_selection = new_model
    
    # 显示当前API状态
    st.write("API状态: " + ("✅ 正常" if st.session_state.nlp_interface.api_working else "❌ 无效或未验证"))

    if st.session_state.data is not None:
        # 确保使用最新数据
        st.session_state.nlp_interface.set_data(st.session_state.data)
        
        # 更新组件
        if 'preprocessor' not in st.session_state or st.session_state.preprocessor is None:
            st.session_state.preprocessor = DataPreprocessor(st.session_state.data)
        st.session_state.nlp_interface.set_preprocessor(st.session_state.preprocessor)
        
        if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
            st.session_state.analyzer = DataAnalyzer(st.session_state.data)
        st.session_state.nlp_interface.set_analyzer(st.session_state.analyzer)
        
        if 'visualizer' not in st.session_state or st.session_state.visualizer is None:
            st.session_state.visualizer = DataVisualizer(st.session_state.data)
        st.session_state.nlp_interface.set_visualizer(st.session_state.visualizer)
    else:
        st.info("请先加载数据，然后您可以在这里提问")
        
    # 显示对话历史
    if len(st.session_state.chat_messages) > 0:
        st.markdown("**对话历史:**")
        chat_container = st.container(height=200, border=True)
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"🧑‍💼: {msg['content']}")
                else:
                    st.markdown(f"🤖: {msg['content']}")

    # 用户输入
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("请输入您的问题", placeholder="例如: 绘制age为纵坐标，survived为横坐标的箱线图")
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("发送")

        if submit_button and user_query:
            # 添加用户消息到历史
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            try:
                # 处理用户查询
                with st.spinner("处理您的请求中..."):
                    response, result = process_user_query(user_query)

                st.success("请求处理完成！")
                
                # 添加AI回复到历史
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                
                # 刷新页面以显示新消息
                # st.rerun()
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"处理失败: {str(e)}\n\n详细信息:\n{error_details}")
                
    # 初始化 session_state 中的变量追踪上一次的选择
    if 'previous_analysis_option' not in st.session_state:
        st.session_state.previous_analysis_option = None
        st.session_state.is_first_load = True

    if "analysis_option" not in st.session_state:
        st.session_state.analysis_option = "默认"
    analysis_option = st.selectbox("分析类型", ["默认", "对话", "文件", "图片", "联网"], index=["默认", "对话", "文件", "图片", "联网"].index(st.session_state.analysis_option))
    st.session_state.analysis_option = analysis_option
    st.write(analysis_option + ("功能：✅ 正常" if st.session_state.nlp_interface.model in model_functions[analysis_option] else "❌ 所选大模型不支持该功能"))

    # 检测选择变化并自动执行相应操作
    if analysis_option != st.session_state.previous_analysis_option and not st.session_state.is_first_load:
        # 首先重置llm功能，确保没有两个功能同时enable
        reset_nlp_interface()

        # 全局设置：除了默认选项，其他选项都同时开启chat_llm_directly
        # st.session_state.nlp_interface.chat_llm_directly = True if analysis_option != "默认" else False

        # 更新上一次选择的值
        prev_option = st.session_state.previous_analysis_option
        st.session_state.previous_analysis_option = analysis_option
        
        if analysis_option == "文件":
            st.session_state.nlp_interface.chat_llm_directly = True
            try:
                # 检查是否有数据
                if st.session_state.data_loader.data is not None:
                    with st.spinner("正在保存数据分析文件..."):
                        # 确定文件类型和扩展名
                        if isinstance(st.session_state.data_loader.data, pd.DataFrame):
                            file_ext = '.csv'
                            file_path = os.path.abspath(f"llm_file{file_ext}")
                            st.session_state.data_loader.data.to_csv(file_path, index=False)
                        else:
                            # 如果不是DataFrame，尝试保存为JSON
                            import json
                            file_ext = '.json'
                            file_path = os.path.abspath(f"llm_file{file_ext}")
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(st.session_state.data_loader.data, f)

                        # 保存文件路径到NLPInterface
                        st.session_state.nlp_interface.local_llm_file_path = file_path
                        st.success(f"数据已保存到: {file_path}")
                else:
                    st.warning("没有可用的数据进行分析，请先加载数据")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"保存数据失败: {str(e)}\n\n详细信息:\n{error_details}")
        elif analysis_option == "图片":
            st.session_state.nlp_interface.chat_llm_directly = True

            # 保存当前图表为图片并更新local_llm_image_path
            if 'current_figure' in st.session_state and st.session_state.current_figure is not None:
                try:
                    with st.spinner("正在保存图表为图片..."):
                        # 创建保存文件路径
                        image_path = os.path.abspath("llm_figure.png")
                        
                        # 设置图表尺寸以控制输出文件大小
                        width = 800  # 图片宽度(像素)
                        height = 600 # 图片高度(像素)
                        scale = 1    # 缩放因子(值越小文件越小)
                        
                        # 保存前更新图表布局
                        fig = st.session_state.current_figure
                        fig.update_layout(
                            width=width,
                            height=height
                        )
                        
                        # 保存图表为图片，指定适当的分辨率
                        fig.write_image(image_path, width=width, height=height, scale=scale)
                        
                        # 检查文件大小
                        file_size = os.path.getsize(image_path) / 1024  # 以KB为单位
                        
                        # 如果文件过大，进一步降低质量重新生成
                        if file_size > 1000:  # 如果大于1MB
                            st.info(f"图表文件较大 ({file_size:.1f}KB)，正在优化大小...")
                            # 使用更低的尺寸和缩放比例
                            width = 600
                            height = 450
                            scale = 0.8
                            fig.update_layout(width=width, height=height)
                            fig.write_image(image_path, width=width, height=height, scale=scale)
                            file_size = os.path.getsize(image_path) / 1024
                        
                        # 更新NLPInterface的图片路径
                        st.session_state.nlp_interface.local_llm_image_path = image_path
                        
                        st.success(f"图表已保存到: {image_path} (大小: {file_size:.1f}KB)")
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"保存图表失败: {str(e)}\n\n详细信息:\n{error_details}")
            else:
                st.warning("没有可用的图表。请先在\"数据可视化\"页面生成一个图表。")
        elif analysis_option == "联网":
            st.session_state.nlp_interface.chat_llm_directly = True
            st.session_state.nlp_interface.enable_llm_internet = True
        elif analysis_option == "对话":
            st.session_state.nlp_interface.chat_llm_directly = True
        elif analysis_option == "默认":
            st.session_state.nlp_interface.chat_llm_directly = False
    elif st.session_state.is_first_load:
        # 初始状态下不开启任何功能
        reset_nlp_interface() 

        # 更新首次加载标记和当前选项
        st.session_state.is_first_load = False
        st.session_state.previous_analysis_option = analysis_option

# 主页面内容
if page == "数据加载":
    st.header("数据加载")
    
    # 加载示例数据
    st.subheader("加载示例数据")
    sample_dataset = st.selectbox(
        "选择示例数据集",
        ["iris", "titanic", "tips"]
    )
    if st.button("加载示例数据"):
        try:
            with st.spinner("正在加载数据..."):
                st.session_state.data = st.session_state.data_loader.load_sample_data(sample_dataset)
                initialize_data_components(st.session_state.data)
            st.success(f"成功加载 {sample_dataset} 数据集")
            st.dataframe(st.session_state.data.head())
        except Exception as e:
            st.error(f"加载数据失败: {str(e)}")
    
    # 上传自定义数据
    st.subheader("上传自定义数据")
    uploaded_file = st.file_uploader("选择文件", type=['csv', 'xlsx', 'json'])
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split('.')[-1]
            with st.spinner("正在加载数据..."):
                st.session_state.data = st.session_state.data_loader.load_data(uploaded_file, file_type)
                initialize_data_components(st.session_state.data)
            st.success("成功加载数据")
            st.dataframe(st.session_state.data.head())
            # 强制刷新页面以确保所有组件都被更新
            # st.rerun()
        except Exception as e:
            st.error(f"加载数据失败: {str(e)}")

elif page == "数据预处理":
    st.header("数据预处理")
    
    if st.session_state.data is None:
        st.warning("请先加载数据")
    else:
        # 每次加载页面时都重新初始化预处理器，确保使用最新数据
        st.session_state.preprocessor = DataPreprocessor(st.session_state.data)
        
        # 数据概览
        st.subheader("数据概览")
        st.write(f"数据集尺寸: {st.session_state.data.shape[0]} 行 × {st.session_state.data.shape[1]} 列")
        st.write("前5行数据预览:")
        st.dataframe(st.session_state.data.head())
        
        # 使用标签页组织各种预处理功能
        preprocess_tabs = st.tabs(["缺失值处理", "异常值处理", "重复值处理", "特征类型转换"])
        
        # 缺失值处理标签页
        with preprocess_tabs[0]:
            # 使用UIComponents显示缺失值处理UI
            new_data = UIComponents.show_missing_values_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="tab_")
            if new_data is not None:
                update_session_data(new_data)
        
        # 异常值处理标签页
        with preprocess_tabs[1]:
            # 使用UIComponents显示异常值处理UI
            new_data = UIComponents.show_outliers_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="tab_")
            if new_data is not None:
                update_session_data(new_data)
        
        # 重复值处理标签页
        with preprocess_tabs[2]:
            # 使用UIComponents显示重复值处理UI
            new_data = UIComponents.show_duplicates_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="tab_")
            if new_data is not None:
                update_session_data(new_data)
        
        # 特征类型转换标签页
        with preprocess_tabs[3]:
            # 使用UIComponents显示特征类型转换UI
            new_data = UIComponents.show_feature_conversion_ui(st.session_state.preprocessor, st.session_state.data, key_prefix="tab_")
            if new_data is not None:
                update_session_data(new_data)

elif page == "数据分析":
    st.header("数据分析")
    
    if st.session_state.data is None:
        st.warning("请先加载数据")
    else:
        # 每次加载页面时都重新初始化分析器，确保使用最新数据
        st.session_state.analyzer = DataAnalyzer(st.session_state.data)
        
        # 描述性统计
        st.subheader("描述性统计")
        
        # 展示特征类型信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**数值型特征**")
            num_features = st.session_state.analyzer.numeric_columns
            if len(num_features) > 0:
                st.write(f"共{len(num_features)}个数值型特征：")
                st.write(", ".join(num_features))
            else:
                st.info("数据集中没有数值型特征")
        
        with col2:
            st.write("**类别型特征**")
            cat_features = st.session_state.analyzer.categorical_columns
            if len(cat_features) > 0:
                st.write(f"共{len(cat_features)}个类别型特征：")
                st.write(", ".join(cat_features))
            else:
                st.info("数据集中没有类别型特征")
        
        # 数值型特征统计
        if len(st.session_state.analyzer.numeric_columns) > 0:
            st.write("### 数值型特征统计")
            # 允许用户选择要分析的数值列
            selected_num_cols = st.multiselect(
                "选择要分析的数值列（默认全选）", 
                options=st.session_state.analyzer.numeric_columns,
                default=list(st.session_state.analyzer.numeric_columns)
            )
            
            if selected_num_cols and st.button("生成数值型特征统计"):
                try:
                    stats = st.session_state.analyzer.get_descriptive_stats(selected_num_cols)
                    st.dataframe(stats)
                except Exception as e:
                    st.error(f"生成统计失败: {str(e)}")
        
        # 类别型特征统计
        if len(st.session_state.analyzer.categorical_columns) > 0:
            st.write("### 类别型特征统计")
            # 允许用户选择要分析的类别列
            selected_cat_cols = st.multiselect(
                "选择要分析的类别列（默认全选）", 
                options=st.session_state.analyzer.categorical_columns,
                default=list(st.session_state.analyzer.categorical_columns)
            )
            
            # 将结果存储到session_state中
            if 'cat_stats' not in st.session_state:
                st.session_state.cat_stats = None
            if 'cat_summary' not in st.session_state:
                st.session_state.cat_summary = None
                
            if selected_cat_cols and st.button("生成类别型特征统计"):
                try:
                    # 计算统计结果并存入session_state
                    st.session_state.cat_stats = st.session_state.analyzer.get_categorical_stats(selected_cat_cols)
                    
                    # 创建汇总表格
                    summary_data = []
                    
                    for col, stats in st.session_state.cat_stats.items():
                        # 获取每个类别变量的唯一值数量
                        unique_count = len(stats)
                        # 获取最常见的类别及其频次
                        most_common_category = stats.index[0]
                        most_common_count = stats.iloc[0]['频次']
                        most_common_percent = stats.iloc[0]['百分比']
                        
                        # 添加到汇总数据中
                        summary_data.append({
                            '特征名称': col,
                            '唯一值数量': unique_count,
                            '最常见类别': most_common_category,
                            '最常见类别频次': most_common_count,
                            '最常见类别占比(%)': most_common_percent
                        })
                    
                    st.session_state.cat_summary = pd.DataFrame(summary_data)
                except Exception as e:
                    st.error(f"生成统计失败: {str(e)}")
            
            # 如果已经计算过统计结果，则显示结果
            if st.session_state.cat_summary is not None:
                # 显示汇总表格
                st.write("**类别型特征统计汇总:**")
                st.dataframe(st.session_state.cat_summary)
                
                # 修改为下拉框选择查看某一个特征的详细分布
                st.write("**类别型特征详细分布:**")
                detail_col = st.selectbox(
                    "选择要查看详细分布的特征",
                    options=selected_cat_cols
                )
                
                if detail_col and st.session_state.cat_stats is not None:
                    if detail_col in st.session_state.cat_stats:
                        st.write(f"**'{detail_col}'的详细分布:**")
                        st.dataframe(st.session_state.cat_stats[detail_col])
                    else:
                        st.warning(f"无法显示'{detail_col}'的详细分布，请重新生成统计数据。")
        
        # 相关性分析
        st.subheader("相关性分析")
        corr_method = st.selectbox(
            "选择相关系数方法",
            ["pearson", "spearman", "kendall"]
        )
        
        # 存储相关性分析结果
        if 'corr_matrix' not in st.session_state:
            st.session_state.corr_matrix = None
            
        if st.button("生成相关性分析"):
            try:
                # 计算相关性矩阵并存储
                st.session_state.corr_matrix = st.session_state.analyzer.correlation_analysis(method=corr_method)
                
                # 显示相关性表格和热力图
                st.write("**相关性系数表格:**")
                # 使用自定义颜色映射函数来实现与热力图相同的配色方案
                def custom_cmap(val):
                    if pd.isna(val):
                        return 'background-color: white'
                    
                    if val == 1:  # 对角线元素
                        return 'background-color: white'
                    elif val >= 0.5:
                        return 'background-color: rgb(180,0,0)'
                    elif val >= 0.2:
                        return 'background-color: rgb(255,100,50)'
                    elif val > 0.001:
                        val_scaled = min(1, val * 10)  # 将小于0.1的值放大
                        return f'background-color: rgba(255,200,150,{val_scaled})'
                    elif val <= -0.5:
                        return 'background-color: rgb(0,0,120)'
                    elif val <= -0.2:
                        return 'background-color: rgb(0,120,255)'
                    elif val < -0.001:
                        val_scaled = min(1, abs(val) * 10)  # 将小于0.1的值放大
                        return f'background-color: rgba(150,220,255,{val_scaled})'
                    else:
                        return 'background-color: rgb(255,255,255)'
                
                # 使用pandas styler映射函数，并保持对角线为白色
                styler = st.session_state.corr_matrix.style.applymap(custom_cmap)
                
                # 显示美化后的表格
                st.dataframe(styler, use_container_width=True)
                
                # 生成热力图
                st.write("**相关性热力图:**")
                
                # 创建热力图，对角线和缺失值为透明
                heatmap = go.Heatmap(
                    z=st.session_state.corr_matrix.values,
                    x=st.session_state.corr_matrix.columns,
                    y=st.session_state.corr_matrix.columns,
                    colorscale=[
                        [0.0, 'rgb(0,0,120)'],      # 深蓝色，强负相关
                        [0.4, 'rgb(0,120,255)'],    # 蓝色，弱负相关
                        [0.45, 'rgb(150,220,255)'], # 淡蓝色，极弱负相关
                        [0.49, 'rgb(220,220,220)'], # 浅灰色，接近无相关
                        [0.5, 'rgb(255,255,255)'],  # 白色，无相关
                        [0.51, 'rgb(220,220,220)'], # 浅灰色，接近无相关
                        [0.55, 'rgb(255,200,150)'], # 淡红橙色，极弱正相关
                        [0.6, 'rgb(255,100,50)'],   # 橙色，弱正相关
                        [1.0, 'rgb(180,0,0)']       # 深红色，强正相关
                    ],
                    zmid=0,                         # 将0设为中间色
                    zmin=-1, 
                    zmax=1,
                    showscale=True,
                    colorbar=dict(
                        title='相关系数',
                        titleside='right',
                        titlefont=dict(size=14),
                        tickvals=[-1, -0.5, -0.1, 0, 0.1, 0.5, 1],
                        ticktext=['-1 (强负相关)', '-0.5', '-0.1', '0', '0.1', '0.5', '1 (强正相关)']
                    )
                )
                
                # 将对角线元素设置为None，实现无色效果
                # 创建一个新的数据矩阵，在对角线位置设为None
                z_matrix = st.session_state.corr_matrix.values.tolist()
                for i in range(len(z_matrix)):
                    z_matrix[i][i] = None
                
                # 更新热力图数据
                heatmap.update(z=z_matrix)
                
                fig = go.Figure()
                fig.add_trace(heatmap)
                
                # 添加图表标题和轴标签
                fig.update_layout(
                    title=f"{corr_method.capitalize()}相关系数热力图",
                    height=600,
                    width=800,
                    xaxis=dict(title='特征', showgrid=False),
                    yaxis=dict(title='特征', showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 添加相关性解释
                st.info("""
                **相关系数解释:**
                - 相关系数范围: -1 到 1
                - 1: 完全正相关（一个变量增加，另一个变量也增加）
                - 0: 无相关性
                - -1: 完全负相关（一个变量增加，另一个变量减少）
                - 颜色越深表示相关性越强
                """)
                
                # 添加强相关变量的提取
                strong_correlations = []
                for i in range(len(st.session_state.corr_matrix.columns)):
                    for j in range(i+1, len(st.session_state.corr_matrix.columns)):
                        if abs(st.session_state.corr_matrix.iloc[i, j]) > 0.5:  # 相关系数绝对值大于0.5视为强相关
                            strong_correlations.append({
                                "变量1": st.session_state.corr_matrix.columns[i],
                                "变量2": st.session_state.corr_matrix.columns[j],
                                "相关系数": round(st.session_state.corr_matrix.iloc[i, j], 3)
                            })
                
                if strong_correlations:
                    st.write("**强相关变量对:**")
                    st.dataframe(pd.DataFrame(strong_correlations))
            except Exception as e:
                st.error(f"生成相关性分析失败: {str(e)}")
        
        # 统计检验
        st.subheader("统计检验")
        test_type = st.selectbox(
            "选择检验类型",
            ["t检验", "方差分析", "卡方检验"]
        )
        
        if test_type == "t检验":
            group_col = st.selectbox("选择分组列", st.session_state.data.columns)
            value_col = st.selectbox("选择数值列", st.session_state.data.select_dtypes(include=['number']).columns)
            group1 = st.selectbox("选择第一组", st.session_state.data[group_col].unique())
            group2 = st.selectbox("选择第二组", st.session_state.data[group_col].unique())
            
            if st.button("进行t检验"):
                try:
                    # 使用最新数据进行测试
                    results = st.session_state.analyzer.t_test(group_col, value_col, group1, group2)
                    
                    # 创建两列布局展示结果
                    col1, col2 = st.columns(2)
                    
                    # 列1：显示统计量
                    with col1:
                        st.subheader("统计量")
                        # 修改数据创建方式，确保Arrow兼容性
                        stats_data = {
                            '统计量': ['t统计量', 'p值', '自由度', f'{group1}均值', f'{group2}均值', '均值差'],
                            '值': [
                                f"{results['t统计量']:.4f}", 
                                f"{results['p值']:.4f}", 
                                int(results['自由度']), 
                                f"{results['组1均值']:.4f}", 
                                f"{results['组2均值']:.4f}", 
                                f"{results['均值差']:.4f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(stats_data))
                    
                    # 列2：显示结果解释
                    with col2:
                        st.subheader("结果解释")
                        
                        # 根据p值设置结果的显示样式
                        if results['p值'] < 0.05:
                            st.success(f"**显著性：** {results['显著性']}")
                        else:
                            st.info(f"**显著性：** {results['显著性']}")
                        
                        st.markdown(f"**详细解释：**")
                        st.markdown(results['结果解释'])
                        
                        # 绘制均值对比图
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[group1, group2],
                            y=[results['组1均值'], results['组2均值']],
                            marker_color=['blue', 'orange']
                        ))
                        fig.update_layout(
                            title=f"{value_col}在不同{group_col}组的均值对比",
                            xaxis_title=group_col,
                            yaxis_title=f"{value_col}的平均值",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"检验失败: {str(e)}")
                    
        elif test_type == "方差分析":
            group_col = st.selectbox("选择分组列", st.session_state.data.columns)
            value_col = st.selectbox("选择数值列", st.session_state.data.select_dtypes(include=['number']).columns)
            
            if st.button("进行方差分析"):
                try:
                    # 进行方差分析
                    results = st.session_state.analyzer.anova_test(group_col, value_col)
                    
                    # 创建两列布局
                    col1, col2 = st.columns(2)
                    
                    # 列1：显示统计量
                    with col1:
                        st.subheader("统计量")
                        stats_data = {
                            '统计量': ['F统计量', 'p值', '组数'],
                            '值': [
                                f"{results['F统计量']:.4f}",
                                f"{results['p值']:.4f}",
                                int(results['组数'])
                            ]
                        }
                        st.dataframe(pd.DataFrame(stats_data))
                    
                    # 列2：显示结果解释和可视化
                    with col2:
                        st.subheader("结果解释")
                        
                        # 根据p值设置结果的显示样式
                        if results['p值'] < 0.05:
                            st.success(f"**显著性：** {results['显著性']}")
                        else:
                            st.info(f"**显著性：** {results['显著性']}")
                        
                        st.markdown(f"**详细解释：**")
                        st.markdown(results['结果解释'])
                    
                    # 显示各组箱线图
                    st.subheader("各组数据分布")
                    try:
                        fig = go.Figure()
                        
                        # 获取所有组
                        groups = st.session_state.data[group_col].unique()
                        
                        # 为每个组添加箱线图
                        for group in groups:
                            group_data = st.session_state.data[st.session_state.data[group_col] == group][value_col]
                            fig.add_trace(go.Box(
                                y=group_data,
                                name=str(group),
                                boxmean=True  # 在箱线图中显示均值
                            ))
                        
                        fig.update_layout(
                            title=f"{value_col}在不同{group_col}组的分布",
                            xaxis_title=group_col,
                            yaxis_title=value_col,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"无法生成箱线图: {str(e)}")
                    
                    # 显示各组均值对比图
                    try:
                        # 计算各组均值
                        group_means = st.session_state.data.groupby(group_col)[value_col].mean().reset_index()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=group_means[group_col].astype(str),
                            y=group_means[value_col],
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10,  # 使用更丰富的配色方案
                            marker=dict(
                                line=dict(width=1, color='#000000')  # 给条形添加细边框
                            ),
                            opacity=0.85,  # 略微透明以增加视觉效果
                            hoverinfo='y+text',
                            text=group_means[value_col].round(2)  # 显示数值
                        ))
                        
                        fig.update_layout(
                            title=f"{value_col}在不同{group_col}组的均值对比",
                            xaxis_title=group_col,
                            yaxis_title=f"{value_col}的平均值",
                            plot_bgcolor='rgba(245, 245, 245, 0.5)',  # 浅灰色背景
                            paper_bgcolor='rgba(255, 255, 255, 1)',   # 白色纸张背景
                            font=dict(size=12),  # 字体大小
                            margin=dict(l=40, r=40, t=60, b=40),  # 边距
                            hoverlabel=dict(bgcolor="white", font_size=12)  # 悬停标签样式
                        )
                        
                        fig.update_yaxes(
                            gridcolor='rgba(200, 200, 200, 0.2)',  # 网格线颜色
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(0, 0, 0, 0.3)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"无法生成均值对比图: {str(e)}")
                    
                except Exception as e:
                    st.error(f"检验失败: {str(e)}")
        
        elif test_type == "卡方检验":
            col1 = st.selectbox("选择第一个类别列", st.session_state.data.select_dtypes(include=['object', 'category']).columns)
            col2 = st.selectbox("选择第二个类别列", st.session_state.data.select_dtypes(include=['object', 'category']).columns)
            
            if st.button("进行卡方检验"):
                try:
                    # 进行卡方检验
                    results = st.session_state.analyzer.chi_square_test(col1, col2)
                    
                    # 创建两列布局
                    col1_ui, col2_ui = st.columns(2)
                    
                    # 列1：显示统计量
                    with col1_ui:
                        st.subheader("统计量")
                        stats_data = {
                            '统计量': ['卡方统计量', 'p值', '自由度'],
                            '值': [
                                f"{results['卡方统计量']:.4f}",
                                f"{results['p值']:.4f}",
                                int(results['自由度'])
                            ]
                        }
                        st.dataframe(pd.DataFrame(stats_data))
                    
                    # 列2：显示结果解释
                    with col2_ui:
                        st.subheader("结果解释")
                        
                        # 根据p值设置结果的显示样式
                        if results['p值'] < 0.05:
                            st.success(f"**显著性：** {results['显著性']}")
                            st.info(f"**关联强度：** {results['关联强度']} (Cramer's V = {results['Cramer_V']:.4f})")
                        else:
                            st.info(f"**显著性：** {results['显著性']}")
                        
                        st.markdown(f"**详细解释：**")
                        st.markdown(results['结果解释'])
                    
                    # 显示列联表
                    st.subheader("列联表")
                    contingency_table = pd.crosstab(
                        st.session_state.data[col1], 
                        st.session_state.data[col2],
                        margins=True,  # 添加行/列总计
                        normalize=False  # 显示频数
                    )
                    st.dataframe(contingency_table)
                    
                    # 显示百分比表
                    st.subheader("百分比表")
                    percentage_table = pd.crosstab(
                        st.session_state.data[col1], 
                        st.session_state.data[col2],
                        normalize='all'  # 计算总体百分比
                    ) * 100
                    
                    # 格式化百分比
                    percentage_table = percentage_table.applymap(lambda x: f"{x:.2f}%")
                    st.dataframe(percentage_table)
                    
                    # 可视化热力图
                    st.subheader("交叉频率热力图")
                    try:
                        # 计算频率表（不包含边际总计）
                        freq_table = pd.crosstab(st.session_state.data[col1], st.session_state.data[col2])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Heatmap(
                            z=freq_table.values,
                            x=freq_table.columns,
                            y=freq_table.index,
                            colorscale='Blues',
                            showscale=True
                        ))
                        
                        fig.update_layout(
                            title=f"{col1}与{col2}的交叉频率热力图",
                            xaxis_title=col2,
                            yaxis_title=col1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"无法生成热力图: {str(e)}")
                    
                except Exception as e:
                    st.error(f"检验失败: {str(e)}")

elif page == "数据可视化":
    st.header("数据可视化")
    
    # 初始化保存图表的全局变量
    if 'current_figure' not in st.session_state:
        st.session_state.current_figure = None
    
    if st.session_state.data is None:
        st.warning("请先加载数据")
    else:
        # 每次加载页面时都重新初始化可视化器，确保使用最新数据
        st.session_state.visualizer = DataVisualizer(st.session_state.data)
        
        # 选择图表类型
        chart_type = st.selectbox(
            "选择图表类型",
            ["条形图", "折线图", "散点图", "箱线图", "小提琴图", "热力图", "直方图", "饼图", "3D散点图", "3D表面图"]
        )
        
        if chart_type == "条形图":
            col1, col2 = st.columns(2)
            with col1:
                x = st.selectbox("选择横坐标(分类变量)", 
                               options=st.session_state.data.columns,
                               help="横坐标通常选择分类变量，将展示每个类别的统计结果")
            with col2:
                y = st.selectbox("选择纵坐标(数值变量)", 
                               options=st.session_state.data.select_dtypes(include=['number']).columns,
                               help="纵坐标需要是数值变量，将计算每个分类的平均值")
            
            # 添加显示数据说明
            data_preview = st.session_state.data.groupby(x)[y].mean().reset_index(drop=True)
            with st.expander("预览聚合数据"):
                st.write("以下是按分类变量聚合后的数据（显示每组平均值）：")
                st.dataframe(data_preview)
            
            color = st.selectbox("选择颜色分组(可选)", [None] + list(st.session_state.data.columns), 
                               help="可选择一个额外的分类变量进行颜色分组")
            
            # 添加聚合方法选择
            agg_method = st.selectbox(
                "选择聚合方法", 
                options=["平均值", "求和", "计数", "最大值", "最小值"],
                help="选择如何聚合每个类别下的数值"
            )
            
            # 聚合方法映射
            agg_method_map = {
                "平均值": "mean",
                "求和": "sum",
                "计数": "count",
                "最大值": "max",
                "最小值": "min"
            }
            
            if st.button("生成条形图"):
                try:
                    with st.spinner("正在生成条形图..."):
                        fig = st.session_state.visualizer.create_bar_chart(
                            x=x, 
                            y=y, 
                            color=color,
                            title=f"{x}与{y}的{agg_method}条形图",
                            agg_method=agg_method_map[agg_method]
                        )
                        # 保存图表到全局变量
                        st.session_state.current_figure = fig
                        st.plotly_chart(fig)
                        
                        # 显示统计信息
                        st.subheader(f"{x}与{y}的统计信息")
                        if color and color != None:
                            # 按两个变量分组统计
                            stats = st.session_state.data.groupby([x, color])[y].agg([
                                agg_method_map[agg_method], 'count', 'std', 'min', 'max'
                            ]).reset_index()
                            # 重命名列
                            stats.columns = [x, color, agg_method, '计数', '标准差', '最小值', '最大值']
                        else:
                            # 按单个变量分组统计
                            stats = st.session_state.data.groupby(x)[y].agg([
                                agg_method_map[agg_method], 'count', 'std', 'min', 'max'
                            ]).reset_index()
                            # 重命名列
                            stats.columns = [x, agg_method, '计数', '标准差', '最小值', '最大值']
                        
                        # 格式化数值列为2位小数
                        for col in stats.columns:
                            if col not in [x, color] and pd.api.types.is_numeric_dtype(stats[col]):
                                stats[col] = stats[col].map(lambda x: f'{x:.2f}' if pd.notnull(x) else x)
                        
                        st.dataframe(stats)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")

        elif chart_type == "散点图":
            x = st.selectbox("选择x轴", st.session_state.data.select_dtypes(include=['number']).columns)
            y = st.selectbox("选择y轴", st.session_state.data.select_dtypes(include=['number']).columns)
            color = st.selectbox("选择颜色分组", [None] + list(st.session_state.data.columns))
            size = st.selectbox("选择大小变量", [None] + list(st.session_state.data.select_dtypes(include=['number']).columns))
            
            if st.button("生成散点图"):
                try:
                    fig = st.session_state.visualizer.create_scatter_plot(x, y, color=color, size=size)
                    # 保存图表到全局变量
                    st.session_state.current_figure = fig
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
        
        elif chart_type == "热力图":
            if st.button("生成热力图"):
                try:
                    fig = st.session_state.visualizer.create_heatmap()
                    # 保存图表到全局变量
                    st.session_state.current_figure = fig
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
                    
        elif chart_type == "折线图":
            x = st.selectbox("选择x轴", st.session_state.data.columns)
            y = st.selectbox("选择y轴", st.session_state.data.select_dtypes(include=['number']).columns)
            color = st.selectbox("选择颜色分组", [None] + list(st.session_state.data.columns))
            
            # 添加排序选项
            sort_x = st.checkbox("按X轴值排序", True, help="对于离散类别数据，建议勾选此选项以确保连线正确")
            
            # 判断x是否为日期类型，如果是，显示相关提示
            is_date = pd.api.types.is_datetime64_any_dtype(st.session_state.data[x]) if x else False
            if is_date:
                st.info("检测到x轴是日期类型，将自动启用时间序列功能。")
            
            # 判断x是否为类别型，提供更多信息
            is_category = (st.session_state.data[x].dtype.name in ['object', 'category']) if x else False
            if is_category:
                st.info(f"检测到x轴'{x}'是类别型数据。对于类别数据，折线图主要用于显示趋势，而不是严格的连续变化。")
            
            # 显示额外的提示，帮助用户做出更好的选择
            if x and y:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**x轴数据类型:**", str(st.session_state.data[x].dtype))
                    n_unique = st.session_state.data[x].nunique()
                    st.write(f"**x轴唯一值数量:** {n_unique}")
                    
                with col2:
                    st.write("**y轴数据类型:**", str(st.session_state.data[y].dtype))
                    if pd.api.types.is_numeric_dtype(st.session_state.data[y]):
                        st.write(f"**y轴数值范围:** {st.session_state.data[y].min():.2f} - {st.session_state.data[y].max():.2f}")
            
            # 检查并提示可能不适合折线图的情况
            if x and y and not is_date and st.session_state.data[x].nunique() < 2:
                st.warning("x轴唯一值数量过少，可能不适合使用折线图。请考虑使用条形图替代。")
            
            if st.button("生成折线图"):
                try:
                    with st.spinner("正在生成折线图..."):
                        # 根据排序选项处理数据
                        if sort_x and not is_date:
                            # 创建数据副本并排序
                            sorted_data = st.session_state.data.sort_values(by=x).copy()
                            # 使用排序后的数据创建可视化器
                            temp_visualizer = DataVisualizer(sorted_data)
                            fig = temp_visualizer.create_line_chart(x, y, color=color)
                        else:
                            fig = st.session_state.visualizer.create_line_chart(x, y, color=color)
                        
                        # 保存图表到全局变量
                        st.session_state.current_figure = fig
                        st.plotly_chart(fig)
                        
                        # 如果存在颜色分组，显示每组的基本统计信息
                        if color and color != None:
                            st.subheader(f"各{color}组的{y}统计信息")
                            group_stats = st.session_state.data.groupby(color)[y].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                            group_stats.columns = [color, '数据点数', '均值', '标准差', '最小值', '最大值']
                            # 格式化数值列为2位小数
                            for col in ['均值', '标准差', '最小值', '最大值']:
                                group_stats[col] = group_stats[col].map(lambda x: f'{x:.2f}')
                            st.dataframe(group_stats)
                            
                            # 添加序列差异分析提示
                            if st.session_state.data[color].nunique() > 1:
                                st.info("提示：您可以在「数据分析 → 统计检验」中使用方差分析或t检验来进一步分析各组之间的差异显著性。")
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
                    # 提供更详细的错误信息和建议
                    if "列 '" in str(e):
                        st.error("请确保所选列在数据集中存在。")
                    elif "类型" in str(e) or "dtype" in str(e):
                        st.error("请确保x轴和y轴的数据类型适合折线图。y轴应为数值类型。")
                    elif "nan" in str(e).lower() or "缺失" in str(e):
                        st.error("所选列中包含缺失值，这可能导致图表生成失败。请先处理缺失值。")
                    else:
                        st.error("请尝试选择不同的列或数据，或查看数据中是否存在异常值。")
                    
        elif chart_type == "箱线图":
            x = st.selectbox("选择x轴(分组)", st.session_state.data.columns)
            y = st.selectbox("选择y轴(数值)", st.session_state.data.select_dtypes(include=['number']).columns)
            
            if st.button("生成箱线图"):
                try:
                    fig = st.session_state.visualizer.create_box_plot(x, y)
                    # 保存图表到全局变量
                    st.session_state.current_figure = fig
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
                    
        elif chart_type == "小提琴图":
            x = st.selectbox("选择x轴(分组)", st.session_state.data.columns)
            y = st.selectbox("选择y轴(数值)", st.session_state.data.select_dtypes(include=['number']).columns)
            
            if st.button("生成小提琴图"):
                try:
                    fig = st.session_state.visualizer.create_violin_plot(x, y)
                    # 保存图表到全局变量
                    st.session_state.current_figure = fig
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
                    
        elif chart_type == "直方图":
            x = st.selectbox("选择数据列", st.session_state.data.select_dtypes(include=['number']).columns)
            color = st.selectbox("选择颜色分组(可选)", [None] + list(st.session_state.data.columns))
            
            if st.button("生成直方图"):
                try:
                    fig = st.session_state.visualizer.create_histogram(x, color=color)
                    # 保存图表到全局变量
                    st.session_state.current_figure = fig
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
                    
        elif chart_type == "饼图":
            names = st.selectbox("选择类别列", st.session_state.data.select_dtypes(include=['object', 'category']).columns)
            values = st.selectbox("选择数值列(可选，默认使用计数)", [None] + list(st.session_state.data.select_dtypes(include=['number']).columns))
            
            if st.button("生成饼图"):
                try:
                    fig = st.session_state.visualizer.create_pie_chart(names, values=values)
                    # 保存图表到全局变量
                    st.session_state.current_figure = fig
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")

        elif chart_type == "3D散点图":
            # 检查是否有足够的数值列
            numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns
            if len(numeric_columns) < 3:
                st.warning("3D散点图需要至少3个数值列。您的数据集中数值列不足。")
            else:
                x = st.selectbox("选择x轴", numeric_columns)
                y = st.selectbox("选择y轴", numeric_columns)
                z = st.selectbox("选择z轴", numeric_columns)
                color = st.selectbox("选择颜色分组(可选)", [None] + list(st.session_state.data.columns))
                size = st.selectbox("选择大小变量(可选)", [None] + list(numeric_columns))
                
                # 显示使用提示
                st.info("提示: 3D图表可以用鼠标拖动旋转，双击恢复默认视角，滚轮缩放。")
                
                if st.button("生成3D散点图"):
                    try:
                        with st.spinner("正在生成3D散点图..."):
                            fig = st.session_state.visualizer.create_3d_scatter(x, y, z, color, size)
                            # 保存图表到全局变量
                            st.session_state.current_figure = fig
                            st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"生成图表失败: {str(e)}")

        elif chart_type == "3D表面图":
            # 检查是否有足够的数值列
            numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns
            if len(numeric_columns) < 3:
                st.warning("3D表面图需要至少3个数值列。您的数据集中数值列不足。")
            else:
                x = st.selectbox("选择x轴", numeric_columns)
                y = st.selectbox("选择y轴", numeric_columns)
                z = st.selectbox("选择z轴(值变量)", numeric_columns)
                
                # 显示使用提示
                st.info("""
                提示: 
                - 3D表面图适合展示三个连续数值变量之间的关系
                - 图表可以用鼠标拖动旋转，双击恢复默认视角，滚轮缩放
                - 对于大数据集，生成表面图可能需要较长时间
                """)
                
                if st.button("生成3D表面图"):
                    try:
                        with st.spinner("正在生成3D表面图..."):
                            fig = st.session_state.visualizer.create_3d_surface(x, y, z)
                            # 保存图表到全局变量
                            st.session_state.current_figure = fig
                            st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"生成图表失败: {str(e)}")
                        if "transparentize" in str(e):
                            st.error("数据量过大或结构不适合生成表面图，请尝试选择不同的变量。")

elif page == "数据建模":
    st.header("📊 数据建模")
    
    if st.session_state.data is None:
        st.warning("请先加载并预处理数据")
    else:
        df = st.session_state.data

        st.subheader("请选择建模类型")
        model_type = st.selectbox("选择模型类别", ["回归分析", "时间序列分析"])

        # ------------------ 回归分析 ------------------ 
        if model_type == "回归分析":
            reg_model = RegressionModel(df)
            reg_subtype = st.radio("选择回归类型", ["线性回归", "逻辑回归", "广义线性回归（含哑变量）"])

            # 动态生成目标变量候选列表
            target_candidates = []
            if reg_subtype == "线性回归":
                target_candidates = df.select_dtypes(include='number').columns.tolist()
            elif reg_subtype == "逻辑回归":
                target_candidates = [col for col in df.columns if df[col].nunique() == 2]
            else:
                target_candidates = df.select_dtypes(include='number').columns.tolist()

            target = st.selectbox("选择目标变量", target_candidates)
            features = st.multiselect("选择特征变量", [col for col in df.columns if col != target])

            # 高级选项
            with st.expander("高级选项", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    scale = st.checkbox("标准化特征", value=True)
                    encode = st.checkbox("编码分类变量", value=True)
                with col2:
                    if reg_subtype == "线性回归":
                        criterion = st.selectbox("模型选择标准", ["AIC", "BIC", "Adj_R2"])
                    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, step=0.05)

            if st.button("训练模型"):
                if not features:
                    st.error("请至少选择一个特征变量")
                else:
                    try:
                        if reg_subtype == "线性回归":
                            with st.spinner("正在训练模型..."):
                                result, diagnostics = reg_model.train_linear(
                                    target=target, 
                                    features=features,
                                    test_size=test_size,
                                    criterion=criterion,
                                    scale=scale,
                                    encode_categorical=encode
                                )

                            st.success("✅ 线性回归模型训练完成")
                            
                            # 模型比较
                            st.subheader("模型比较")
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("全模型AIC", f"{result['full_metrics']['AIC']:.1f}")
                                st.metric("全模型BIC", f"{result['full_metrics']['BIC']:.1f}")
                            with cols[1]:
                                st.metric("最优子集AIC", f"{result['best_metrics']['AIC']:.1f}", 
                                         delta=result['full_metrics']['AIC']-result['best_metrics']['AIC'])
                                st.metric("最优子集BIC", f"{result['best_metrics']['BIC']:.1f}",
                                         delta=result['full_metrics']['BIC']-result['best_metrics']['BIC'])
                            with cols[2]:
                                st.metric("调整R²", f"{result['best_metrics']['Adj_R2']:.3f}")
                                st.metric("测试集R²", f"{result['test_r2']:.3f}")
                            
                            st.write("**入选特征:**", ", ".join(result['best_subset']))

                            # 诊断图表
                            st.subheader("模型诊断")
                            tab1, tab2, tab3 = st.tabs(["残差诊断", "共线性分析", "模型摘要"])
                            with tab1:
                                st.pyplot(diagnostics['plot'])
                            with tab2:
                                st.dataframe(
                                    diagnostics['vif'].style.highlight_between(
                                        subset=['VIF'], 
                                        color='lightcoral',
                                        left=5,  # VIF>5表示可能有共线性
                                        right=float('inf')
                                    ).format(precision=2)
                                )
                                st.info("VIF>5表示可能存在共线性问题")
                            with tab3:
                                st.text(str(result['model'].summary()))

                        elif reg_subtype == "逻辑回归":
                            # 编码目标变量
                            df[target] = df[target].astype('category').cat.codes
                            reg_model = RegressionModel(df)
                            result = reg_model.train_logistic(
                                target=target,
                                features=features,
                                test_size=test_size,
                                scale=scale,
                                encode_categorical=encode
                            )
                            
                            st.success("✅ 逻辑回归模型训练完成")
                            
                            # 分类指标
                            st.subheader("分类指标")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("准确率", f"{result['metrics']['accuracy']:.3f}")
                            col2.metric("精确率", f"{result['metrics']['precision']:.3f}")
                            col3.metric("召回率", f"{result['metrics']['recall']:.3f}")
                            col4.metric("F1分数", f"{result['metrics']['f1']:.3f}")

                            # 混淆矩阵
                            st.subheader("混淆矩阵")
                            cm = confusion_matrix(result['y_test'], result['y_pred'])
                            fig_cm = px.imshow(
                                cm,
                                labels=dict(x="预测", y="真实", color="数量"),
                                x=['0', '1'], 
                                y=['0', '1'],
                                text_auto=True,
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)

                            # ROC曲线
                            st.subheader("ROC曲线")
                            fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
                            roc_auc = auc(fpr, tpr)
                            fig_roc = go.Figure()
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC曲线 (AUC = {roc_auc:.2f})',
                                line=dict(color='darkorange', width=2)
                            ))
                            fig_roc.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                line=dict(dash='dash', color='navy'),
                                name='随机猜测'
                            ))
                            fig_roc.update_layout(
                                title='接收者操作特征曲线',
                                xaxis_title='假正率',
                                yaxis_title='真正率',
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_roc, use_container_width=True)

                        else:  # 广义线性模型
                            glm_family = st.selectbox("选择GLM族", ["Gaussian", "Poisson", "Binomial"])
                            glm_family_map = {
                                "Gaussian": sm.families.Gaussian(),
                                "Poisson": sm.families.Poisson(),
                                "Binomial": sm.families.Binomial()
                            }
                            result = reg_model.train_glm(
                                target=target,
                                features=features,
                                family=glm_family_map[glm_family]
                            )
                            st.success("✅ 广义线性模型拟合完成")
                            st.subheader("模型摘要")
                            st.text(result["summary"])
                            st.write("模型系数:")
                            st.dataframe(result["coefficients"].to_frame().style.format(precision=3))

                    except Exception as e:
                        st.error(f"❌ 模型训练失败：{str(e)}")
                        if "singular matrix" in str(e):
                            st.info("建议：检查特征间是否存在完全共线性，或尝试减少特征数量")

        # ------------------ 时间序列分析 ------------------ 
        elif model_type == "时间序列分析":
            ts_model = TimeSeriesModel(df)

            # 界面布局
            col1, col2 = st.columns([2, 1])
            with col1:
                date_col = st.selectbox("选择时间列", df.select_dtypes(include=['datetime', 'object']).columns)
                value_col = st.selectbox("选择分析列", df.select_dtypes(include='number').columns)
                
            with col2:
                train_ratio = st.slider("训练集比例", 0.5, 0.95, 0.8, step=0.05)
                forecast_steps = st.number_input("预测步数", 1, 365, 30)
                ci_level = st.slider("置信区间", 0.8, 0.99, 0.95)

            if st.button("运行分析"):
                try:
                    # 数据预处理
                    df[date_col] = pd.to_datetime(df[date_col])
                    ts_series = df.set_index(date_col)[value_col].asfreq('D').ffill()

                    # 划分训练测试集
                    train_size = int(len(ts_series) * train_ratio)
                    train, test = ts_series[:train_size], ts_series[train_size:]

                    # 模型训练
                    result = ts_model.full_analysis(train, max_diff=2)
                    forecast = ts_model.forecast(steps=forecast_steps)

                    # 可视化
                    fig = go.Figure()
                    # 历史数据
                    fig.add_trace(go.Scatter(
                        x=train.index, 
                        y=train,
                        mode='lines',
                        name='训练数据',
                        line=dict(color='#1f77b4')
                    ))
                    
                    # 测试数据（如果有）
                    if len(test) > 0:
                        fig.add_trace(go.Scatter(
                            x=test.index,
                            y=test,
                            mode='lines',
                            name='实际值',
                            line=dict(color='green')
                        ))

                    # 预测结果
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast['mean'],
                        mode='lines+markers',
                        name='预测均值',
                        line=dict(color='firebrick', width=2)
                    ))
                    
                    # 置信区间
                    fig.add_trace(go.Scatter(
                        x=forecast.index.tolist() + forecast.index[::-1].tolist(),
                        y=forecast['mean_ci_upper'].tolist() + forecast['mean_ci_lower'][::-1].tolist(),
                        fill='toself',
                        fillcolor='rgba(255,165,0,0.2)',
                        line=dict(color='rgba(255,165,0,0.1)'),
                        name=f'{ci_level*100:.0f}% 置信区间'
                    ))

                    fig.update_layout(
                        title=f'{value_col} 时间序列预测',
                        xaxis_title='日期',
                        yaxis_title=value_col,
                        hovermode='x unified',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 模型评估
                    st.subheader("模型评估")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("最优模型参数", f"ARIMA{result['order']}")
                        st.metric("AIC", f"{result['model'].aic:.1f}")
                    with col2:
                        st.metric("训练数据量", f"{len(train)} 条")
                        st.metric("预测区间", f"{forecast.index[0].date()} ~ {forecast.index[-1].date()}")

                    # 残差诊断
                    st.subheader("残差诊断")
                    residuals = result['model'].resid
                    fig_res = make_subplots(rows=2, cols=1)
                    fig_res.add_trace(go.Scatter(
                        y=residuals,
                        mode='lines',
                        name='残差序列'
                    ), row=1, col=1)
                    fig_res.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=30,
                        name='残差分布'
                    ), row=2, col=1)
                    fig_res.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig_res, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 分析失败：{str(e)}")
                    if "non-stationary" in str(e):
                        st.info("提示：请尝试更高的差分阶数或进行数据转换")
elif page == "SQL数据库":
    st.header("SQL数据库")
    
    # 检查数据库连接是否可用
    if st.session_state.db_manager is None:
        st.error("无法连接到数据库。请检查连接设置并重试。")
        
        # 提供数据库连接设置
        with st.expander("数据库连接设置", expanded=True):
            db_type = st.selectbox("数据库类型", 
                                 options=["auto", "mysql", "sqlite"], 
                                 format_func=lambda x: {
                                     "auto": "自动选择 (推荐)",
                                     "mysql": "MySQL", 
                                     "sqlite": "SQLite (无需服务器)"
                                 }[x])
            
            if db_type in ["auto", "mysql"]:
                db_host = st.text_input("数据库主机", value="localhost")
                db_user = st.text_input("用户名", value="root")
                db_password = st.text_input("密码", type="password", value="password")
            else:
                st.info("SQLite数据库无需服务器，将在本地文件中存储数据")
                db_host = "localhost"
                db_user = "root"
                db_password = "password"
                
            db_name = st.text_input("数据库名", value="data_analysis", 
                                  help="MySQL模式: 数据库名称, SQLite模式: 将创建同名的.db文件")
            
            if st.button("连接数据库"):
                try:
                    with st.spinner("正在连接数据库..."):
                        st.session_state.db_manager = DBManager(
                            host=db_host,
                            user=db_user,
                            password=db_password,
                            database=db_name,
                            db_type=db_type
                        )
                    
                    # 获取连接信息
                    conn_info = st.session_state.db_manager.get_connection_info()
                    db_type_display = conn_info.get("数据库类型", "未知")
                    
                    st.success(f"成功连接到{db_type_display}数据库")
                    # st.rerun()
                except Exception as e:
                    st.error(f"数据库连接失败: {str(e)}")
                    if "auto" in db_type and "mysql" in str(e).lower():
                        st.info("提示: 如果您没有运行MySQL服务器，请选择'SQLite'数据库类型")
    else:
        # 数据库已连接，显示连接信息
        conn_info = st.session_state.db_manager.get_connection_info()
        db_type_display = conn_info.get("数据库类型", "未知")
        
        # 在侧边栏或顶部显示数据库类型
        st.info(f"当前使用: {db_type_display} 数据库")
        
        # 将当前数据加载到数据库
        st.subheader("将当前数据加载到数据库")
        
        if st.session_state.data is None:
            st.warning("请先在「数据加载」页面加载数据")
        else:
            # 显示数据预览
            st.write("当前加载的数据预览:")
            st.dataframe(st.session_state.data.head())
            
            # 生成默认表名
            default_table_name = ""
            if hasattr(st.session_state.data_loader, 'current_file_name') and st.session_state.data_loader.current_file_name:
                # 使用文件名作为默认表名
                default_table_name = os.path.splitext(st.session_state.data_loader.current_file_name)[0]
            else:
                # 使用日期时间作为默认表名
                default_table_name = f"table_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            # 允许用户指定表名
            table_name = st.text_input("表名", value=default_table_name)
            
            if st.button("加载到数据库"):
                try:
                    with st.spinner("正在加载数据到数据库..."):
                        final_table_name = st.session_state.db_manager.load_dataframe_to_db(
                            st.session_state.data, 
                            table_name
                        )
                        st.session_state.current_db_table = final_table_name
                        
                        # 更新表列表
                        st.session_state.db_tables = st.session_state.db_manager.get_all_tables()
                        
                        st.success(f"数据已成功加载到表 `{final_table_name}`")
                        
                        # 显示表结构
                        table_info = st.session_state.db_manager.get_table_info(final_table_name)
                        st.subheader("表结构")
                        st.dataframe(table_info)
                        
                except Exception as e:
                    st.error(f"加载数据到数据库失败: {str(e)}")
        
        # 显示分隔线
        st.markdown("---")
        
        # SQL查询区域
        st.subheader("SQL查询")
        
        # 显示当前可用的表
        try:
            tables = st.session_state.db_manager.get_all_tables()
            st.session_state.db_tables = tables
            
            if not tables:
                st.info("数据库中暂无表，请先加载数据到数据库")
            else:
                # 显示可用表列表
                st.write("可用表:")
                table_cols = st.columns(min(4, len(tables)))
                for i, table in enumerate(tables):
                    with table_cols[i % len(table_cols)]:
                        st.code(table)
                
                # 如果有当前表，显示表信息
                if st.session_state.current_db_table and st.session_state.current_db_table in tables:
                    with st.expander(f"表 {st.session_state.current_db_table} 的结构"):
                        table_info = st.session_state.db_manager.get_table_info(st.session_state.current_db_table)
                        st.dataframe(table_info)
                
                # SQL查询输入
                default_query = ""
                if st.session_state.current_db_table:
                    default_query = f"SELECT * FROM {st.session_state.current_db_table} LIMIT 10"
                
                query = st.text_area("输入SQL查询", value=default_query, height=150)
                
                # 查询历史下拉框
                if st.session_state.sql_history:
                    history_choice = st.selectbox(
                        "从历史查询中选择", 
                        options=[""] + st.session_state.sql_history,
                        format_func=lambda x: x[:50] + "..." if len(x) > 50 else x
                    )
                    if history_choice and history_choice != query:
                        query = history_choice
                        st.experimental_rerun()
                
                if st.button("执行查询"):
                    if not query.strip():
                        st.warning("请输入SQL查询")
                    else:
                        try:
                            with st.spinner("执行查询中..."):
                                result, message = st.session_state.db_manager.execute_query(query)
                                
                                # 将查询添加到历史记录
                                if query not in st.session_state.sql_history:
                                    st.session_state.sql_history.insert(0, query)
                                    # 限制历史记录长度
                                    if len(st.session_state.sql_history) > 10:
                                        st.session_state.sql_history = st.session_state.sql_history[:10]
                                
                                # 显示执行结果
                                st.success(message)
                                
                                if not result.empty:
                                    st.subheader("查询结果")
                                    st.dataframe(result)
                                    
                                    # 显示下载按钮
                                    download_formats = {
                                        "CSV": result.to_csv(index=False).encode('utf-8'),
                                        "Excel": None,  # 将在下面正确处理
                                        "JSON": result.to_json(orient='records', force_ascii=False)
                                    }
                                    
                                    download_format = st.selectbox("选择下载格式", list(download_formats.keys()))
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    extensions = {"CSV": "csv", "Excel": "xlsx", "JSON": "json"}
                                    filename = f"query_result_{timestamp}.{extensions[download_format]}"
                                    
                                    # 特殊处理Excel格式
                                    if download_format == "Excel":
                                        # 创建BytesIO对象来存储Excel数据
                                        excel_buffer = io.BytesIO()
                                        # 将DataFrame写入BytesIO对象
                                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                            result.to_excel(writer, index=False)
                                        # 获取数据
                                        excel_data = excel_buffer.getvalue()
                                        st.download_button(
                                            label="下载结果",
                                            data=excel_data,
                                            file_name=filename,
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    else:
                                        st.download_button(
                                            label="下载结果",
                                            data=download_formats[download_format],
                                            file_name=filename,
                                            mime={"CSV": "text/csv", "JSON": "application/json"}[download_format]
                                        )
                        except Exception as e:
                            st.error(f"执行查询失败: {str(e)}")
        except Exception as e:
            st.error(f"获取表信息失败: {str(e)}")
            with st.expander("错误详情"):
                st.code(str(e))

# 对话结果显示区域
if 'ai_result' in st.session_state:
    # 处理返回的结果
    result = st.session_state.ai_result

    # 显示分析结果
    # 只有在有结果时才显示
    if result:
        # 显示错误信息
        if 'error' in result:
            st.error(result.get('message', "处理请求时出错"))
            if 'suggestion' in result:
                st.info(result['suggestion'])

        # 如果有SQL生成结果
        if 'sql' in result:
            st.header("🔍 SQL生成结果")
            st.code(result['sql'], language="sql")
            # 复制按钮
            if st.button("复制到SQL查询编辑器"):
                st.session_state.sql_query = result['sql']
                # st.rerun()

        # 显示智能助手分析结果
        if st.session_state.data is not None and any(k in result for k in ['show_missing_values_ui', 'show_duplicates_ui', 'show_feature_conversion_ui',
                                   'show_descriptive_stats_ui', 'show_correlation_ui', 'show_statistical_tests_ui',
                                   'show_outliers_ui', 'charts', 'html']):
            st.header("💬 智能助手分析结果")

            # 展示数据信息
            if 'info' in result:
                with st.expander("数据信息", expanded=False):
                    st.dataframe(pd.DataFrame(result['info']).T)

            # 展示数据预览
            if 'preview' in result:
                with st.expander("数据预览", expanded=False):
                    preview_df = pd.DataFrame(result['preview'])
                    st.dataframe(preview_df)

            # 根据结果类型显示不同的UI组件
            if 'show_missing_values_ui' in result and result['show_missing_values_ui']:
                show_ui_component('missing_values', result)

            if 'show_duplicates_ui' in result and result['show_duplicates_ui']:
                show_ui_component('duplicates', result)

            if 'show_feature_conversion_ui' in result and result['show_feature_conversion_ui']:
                show_ui_component('feature_conversion', result)

            if 'show_descriptive_stats_ui' in result and result['show_descriptive_stats_ui']:
                show_ui_component('descriptive_stats', result)

            if 'show_correlation_ui' in result and result['show_correlation_ui']:
                show_ui_component('correlation', result)

            if 'show_statistical_tests_ui' in result and result['show_statistical_tests_ui']:
                show_ui_component('statistical_test', result)

            if 'show_outliers_ui' in result and result['show_outliers_ui']:
                show_ui_component('outliers', result)

            # 展示图表
            if 'charts' in result:
                charts = result['charts']
                if charts:
                    st.subheader("📊 数据可视化")
                    for chart in charts:
                        if isinstance(chart, dict) and 'html' in chart:
                            components.html(chart['html'], height=chart.get('height', 600))
                        elif isinstance(chart, str) and chart.startswith('<'):
                            components.html(chart, height=600)
                        else:
                            st.warning(f"未知的图表格式: {type(chart)}")

            # 展示HTML内容
            if 'html' in result:
                components.html(result['html'], height=600)

        # 显示智谱AI的直接回答 - 移到外部，无论是否有数据都显示
        if 'response_directly' in result and result['response_directly']:
            # 获取当前模型名称
            model_display_names = {
                "azureopenai": "OpenAI",
                "qwen": "通义AI",
                "deepseek": "DeepSeek",
                "zhipu": "智谱AI",
                "doubao": "豆包AI",
                "localmodel": "本地模型"
            }
            current_model = st.session_state.nlp_interface.model
            model_display_name = model_display_names.get(current_model, "AI")
            st.subheader(f"📝 {model_display_name}的直接回答")
            st.markdown(result['response_directly'])
