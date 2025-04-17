"""
Amazon关键词智能分析系统 - FastAPI Web应用
版本3.5
功能: 关键词聚类分析 + 大模型智能标签生成 + Web接口
"""

import os
import logging
import glob
import time
import random
import shutil
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import sparse
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 初始化环境变量
load_dotenv()

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("keyword_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("static", exist_ok=True)

# 定义数据模型
class AnalysisRequest(BaseModel):
    job_id: Optional[str] = None
    cluster_method: str = "kmeans" 
    num_clusters: int = 10
    threshold: float = 0.3
    use_weights: bool = True
    min_df: int = 2
    max_df: float = 0.95

class AnalysisStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result_file: Optional[str] = None

# 任务状态跟踪
analysis_tasks = {}

class KeywordAnalyzer:
    """关键词分析核心类"""
    
    # 配置参数
    DEFAULT_COLUMNS = [
        '关键词', '关键词翻译', '流量占比', '预估周曝光量', 'ABA周排名', 
        '月搜索量', '月购买量', '购买率', '展示量', '点击量', 'SPR', 
        '商品数', '需供比', '广告竞品数', '点击集中度', '前三ASIN转化总占比', 'PPC竞价', '相关产品'
    ]
    NUMERIC_COLUMNS = [
        '流量占比', '预估周曝光量', '月搜索量', '月购买量', '购买率', 
        '展示量', '点击量', 'SPR', '商品数', '需供比', '广告竞品数', 
        '点击集中度', '前三ASIN转化总占比', 'PPC竞价', '相关产品'
    ]
    WEIGHT_COLUMNS = ['流量占比', '月搜索量', '购买率']
    
    MAX_SAMPLE = 100      # 最大抽样关键词数
    MAX_CHARS = 2000      # 最大样本字符数
    MIN_SAMPLE = 20       # 最小保留样本数
    TIMEOUT = 30          # API超时时间(秒)
    MAX_RETRIES = 3       # API最大重试次数

    def __init__(self, job_id: str, params: AnalysisRequest):
        """初始化分析器"""
        self.job_id = job_id
        self.cluster_method = params.cluster_method
        self.num_clusters = params.num_clusters
        self.threshold = params.threshold
        self.use_weights = params.use_weights
        self.min_df = params.min_df
        self.max_df = params.max_df
        
        self._validate_env()
        self.client = ZhipuAI(api_key=os.getenv('ZHIPUAI_API_KEY'))
        self.model_name = os.getenv('ZHIPUAI_MODEL_NAME', 'glm-4-flash')
        self.df = None
        
        # 设置NLTK数据目录
        nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk'))
        self.stemmer = PorterStemmer()
        
        # 进度追踪
        self.progress = 0.0
        self.status_message = "初始化中..."
        self.update_status()

    def _validate_env(self):
        """验证环境配置"""
        if not os.getenv('ZHIPUAI_API_KEY'):
            logger.error("未找到ZHIPUAI_API_KEY,请检查.env文件")
            raise ValueError("缺少API密钥配置")
            
    def update_status(self, progress=None, message=None):
        """更新任务状态"""
        if progress is not None:
            self.progress = min(1.0, max(0.0, progress))
        if message is not None:
            self.status_message = message
            logger.info(f"任务 {self.job_id}: {message} ({self.progress*100:.1f}%)")
            
        analysis_tasks[self.job_id] = {
            "status": "processing" if self.progress < 1.0 else "completed",
            "progress": self.progress,
            "message": self.status_message,
            "result_file": f"results/{self.job_id}.xlsx" if self.progress >= 1.0 else None
        }

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载Excel数据"""
        try:
            self.update_status(0.05, "正在加载数据...")
            
            # 保留第一个sheet，不使用默认标题行
            xls = pd.ExcelFile(file_path)
            df = pd.read_excel(xls, sheet_name=0, header=None)
            
            # 删除第一行
            df = df.iloc[1:].reset_index(drop=True)
            
            # 重新设置列名
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            
            self.update_status(0.1, f"成功加载数据：{len(df)}行")
            return df
        except Exception as e:
            logger.error(f"数据加载失败：{str(e)}")
            self.update_status(message=f"数据加载失败：{str(e)}")
            raise

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        self.update_status(0.15, "正在进行数据预处理...")
        
        # 智能处理空值
        for col in df.columns:
            if col in self.NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 数值列用0填充
                df[col] = df[col].fillna(0)
            else:
                # 非数值列用空字符串填充
                df[col] = df[col].fillna('')
        
        # 移除"PPC竞价"列中的货币符号
        if 'PPC竞价' in df.columns:
            df['PPC竞价'] = df['PPC竞价'].astype(str).replace('[\$,]', '', regex=True)
            df['PPC竞价'] = pd.to_numeric(df['PPC竞价'], errors='coerce').fillna(0)
        
        # 筛选"相关产品"列的值为最大数值的threshold比例
        if '相关产品' in df.columns:
            threshold_value = df['相关产品'].max() * self.threshold
            df = df[df['相关产品'] >= threshold_value]
            self.update_status(message=f"筛选后保留{len(df)}行数据")
            
        # 按"流量占比"降序排列
        if '流量占比' in df.columns:
            df = df.sort_values(by='流量占比', ascending=False)
        
        # 检测并规范化数值特征
        numeric_cols = [col for col in self.NUMERIC_COLUMNS if col in df.columns]
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logger.info(f"已对{len(numeric_cols)}个数值特征进行标准化")
        
        # 保留需要的列
        valid_cols = [c for c in self.DEFAULT_COLUMNS if c in df.columns]
        if len(valid_cols) < 3:
            logger.warning("数据列不完整，使用所有可用列")
            return df
            
        self.update_status(0.2, "数据预处理完成")
        return df[valid_cols]

    def _stem_text(self, text: str) -> str:
        """文本预处理：小写化+词干提取"""
        try:
            words = word_tokenize(str(text).lower())
            return ' '.join([self.stemmer.stem(w) for w in words])
        except Exception as e:
            logger.warning(f"词干提取失败: {e}")
            return str(text).lower()

    def _select_samples(self, keywords: List[str]) -> List[str]:
        """选择代表性样本"""
        if len(keywords) <= self.MAX_SAMPLE:
            return keywords
        
        selected = random.sample(keywords, self.MAX_SAMPLE)
        total_len = sum(len(str(s)) for s in selected)
        
        while total_len > self.MAX_CHARS and len(selected) > self.MIN_SAMPLE:
            selected.pop()
            total_len = sum(len(str(s)) for s in selected)
        
        return selected

    def _generate_cluster_label(self, keywords: List[str], cluster_id: int) -> str:
        """生成聚类标签"""
        samples = self._select_samples(keywords)
        prompt = f"""
        以下是一组相关的关键词（共{len(keywords)}个，显示{len(samples)}个样本）：
        {', '.join(map(str, samples))}

        请根据这些关键词的共同特征，为它们创建一个简洁而准确的分类标签。
        这个标签应该：
        1. 反映这组关键词的核心主题或共同点
        2. 不超过5个字
        3. 尽可能具体和有描述性
        4. 可以是产品类别、功能特征、使用场景、用户需求等任何相关的分类

        请直接给出分类标签，无需其他解释。
        """
        
        for attempt in range(1, self.MAX_RETRIES+1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.TIMEOUT,
                    stream=True,
                )
                
                label = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        label += chunk.choices[0].delta.content
                
                label = label.strip()
                
                # 限制标签长度
                if len(label) > 5:
                    label = label[:5]
                    
                return label
            except Exception as e:
                logger.warning(f"聚类{cluster_id}标签生成失败（尝试{attempt}/{self.MAX_RETRIES}):{str(e)}")
                time.sleep(2**attempt)
        
        return f"聚类{cluster_id}"

    def _find_optimal_k(self, X, max_k=15):
        """使用轮廓系数寻找最佳聚类数"""
        self.update_status(0.3, "正在寻找最佳聚类数...")
        
        # 如果数据量较小，调整最大k值
        n_samples = X.shape[0]
        max_k = min(max_k, n_samples - 1)
        
        if max_k <= 1:
            return min(self.num_clusters, n_samples)
            
        # 尝试不同的k值
        silhouette_scores = []
        k_values = range(2, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(score)
            
        # 找到最佳k值
        best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        logger.info(f"最佳聚类数: {best_k}")
        
        return best_k

    def analyze(self, file_path: str):
        """执行完整分析流程"""
        try:
            # 数据准备
            self.df = self.load_data(file_path)
            self.df = self.preprocess(self.df)
            
            if len(self.df) == 0:
                self.update_status(1.0, "筛选后数据为空，无法进行聚类")
                return
            
            # 文本处理
            self.update_status(0.25, "正在处理关键词文本...")
            self.df['处理文本'] = self.df['关键词'].apply(self._stem_text)
            
            # 向量化
            vectorizer = TfidfVectorizer(min_df=self.min_df, max_df=self.max_df, max_features=1000)
            X_text = vectorizer.fit_transform(self.df['处理文本'])
            
            # 准备要使用的特征
            X = X_text
            
            # 应用关键词权重
            if self.use_weights:
                self.update_status(0.28, "正在应用关键词权重...")
                weight_metrics = [m for m in self.WEIGHT_COLUMNS if m in self.df.columns]
                
                if weight_metrics:
                    weights = self.df[weight_metrics].copy()
                    for metric in weight_metrics:
                        min_val = weights[metric].min()
                        max_val = weights[metric].max()
                        if min_val != max_val:
                            weights[metric] = 0.1 + 0.9 * (weights[metric] - min_val) / (max_val - min_val)
                        else:
                            weights[metric] = 1.0
                    
                    final_weights = weights.mean(axis=1).values
                    weight_matrix = sparse.diags(final_weights)
                    X = weight_matrix.dot(X_text)
                    logger.info(f"已应用{len(weight_metrics)}个指标作为关键词权重")
            
            # 降维处理（对于大数据集）
            if X.shape[0] > 1000 and X.shape[1] > 50:
                self.update_status(0.32, "正在进行降维处理...")
                pca = PCA(n_components=min(50, X.shape[1]-1))
                X_dense = X.toarray() if sparse.issparse(X) else X
                X = pca.fit_transform(X_dense)
                logger.info(f"PCA降维: {X_dense.shape} -> {X.shape}")
            
            # 自动确定最佳聚类数（如果需要）
            n_clusters = self.num_clusters
            if n_clusters <= 0:
                n_clusters = self._find_optimal_k(X)
            n_clusters = min(n_clusters, len(self.df))
            
            # 聚类分析
            self.update_status(0.35, f"正在进行{self.cluster_method}聚类(k={n_clusters})...")
            
            if self.cluster_method.lower() == 'dbscan':
                # DBSCAN聚类
                epsilon = 0.5  # 可以根据数据特性调整
                min_samples = 5  # 每个聚类至少的样本数
                dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
                self.df['原始聚类编号'] = dbscan.fit_predict(X)
                
                # DBSCAN会将噪声点标记为-1，我们将这些点分到一个单独的聚类
                if -1 in self.df['原始聚类编号'].values:
                    noise_cluster = self.df['原始聚类编号'].max() + 1
                    self.df.loc[self.df['原始聚类编号'] == -1, '原始聚类编号'] = noise_cluster
            else:
                # 默认使用K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.df['原始聚类编号'] = kmeans.fit_predict(X)
            
            # 生成标签
            self.update_status(0.4, "正在生成聚类标签...")
            cluster_labels = {}
            unique_clusters = self.df['原始聚类编号'].unique()
            total_clusters = len(unique_clusters)
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for i, cid in enumerate(unique_clusters):
                    cluster_keywords = self.df[self.df['原始聚类编号'] == cid]['关键词'].tolist()
                    futures[executor.submit(self._generate_cluster_label, cluster_keywords, cid)] = cid
                
                for i, future in enumerate(futures):
                    cid = futures[future]
                    try:
                        cluster_labels[cid] = future.result()
                        # 更新标签生成进度
                        label_progress = 0.4 + 0.3 * ((i + 1) / total_clusters)
                        self.update_status(label_progress, f"已生成{i+1}/{total_clusters}个标签")
                    except Exception as e:
                        logger.error(f"标签生成失败: {e}")
                        cluster_labels[cid] = f"聚类{cid}"
            
            # 应用标签
            self.df['关键词分类'] = self.df['原始聚类编号'].map(cluster_labels)
            
            # 计算每个聚类的流量占比总和
            self.update_status(0.75, "正在整理分析结果...")
            if '流量占比' in self.df.columns:
                cluster_traffic = self.df.groupby('原始聚类编号')['流量占比'].sum().sort_values(ascending=False)
                
                # 创建新的聚类编号映射，按流量占比排序
                new_cluster_numbers = {old: new+1 for new, old in enumerate(cluster_traffic.index)}
                
                # 更新DataFrame中的聚类编号
                self.df['聚类编号'] = self.df['原始聚类编号'].map(new_cluster_numbers)
            else:
                # 如果没有流量占比，简单地重新编号
                self.df['聚类编号'] = self.df['原始聚类编号'] + 1
            
            # 保存结果
            self.update_status(0.9, "正在保存结果...")
            result_file = self._save_results()
            
            self.update_status(1.0, "分析完成")
            return result_file
            
        except Exception as e:
            logger.error(f"分析失败：{str(e)}")
            self.update_status(1.0, f"分析失败：{str(e)}")
            raise

    def _save_results(self):
        """保存分析结果"""
        # 重新排序列
        output_cols = ['关键词', '聚类编号', '关键词分类', '流量占比']
        remaining_cols = [c for c in self.df.columns if c not in output_cols + ['原始聚类编号', '处理文本']]
        output_cols += remaining_cols
        
        # 确保所有列都存在
        output_cols = [c for c in output_cols if c in self.df.columns]
        
        # 保存Excel格式
        result_file = f"results/{self.job_id}.xlsx"
        self.df[output_cols].to_excel(result_file, index=False)
        
        logger.info(f"结果已保存至{result_file}，共{len(self.df)}条数据")
        return result_file

# 创建FastAPI应用
app = FastAPI(
    title="Amazon关键词智能分析系统",
    description="提供关键词聚类分析和智能标签生成服务",
    version="3.5"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

def analyze_background(file_path: str, job_id: str, params: AnalysisRequest):
    """后台执行分析任务"""
    try:
        analyzer = KeywordAnalyzer(job_id, params)
        analyzer.analyze(file_path)
    except Exception as e:
        logger.error(f"任务 {job_id} 执行失败: {str(e)}")
        analysis_tasks[job_id] = {
            "status": "failed",
            "progress": 1.0,
            "message": f"分析失败: {str(e)}",
            "result_file": None
        }

@app.post("/api/analyze", response_model=AnalysisStatus)
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    cluster_method: str = Query("kmeans", description="聚类方法: kmeans 或 dbscan"),
    num_clusters: int = Query(10, description="聚类数量 (K-means)"),
    threshold: float = Query(0.3, description="相关产品筛选阈值"),
    use_weights: bool = Query(True, description="是否使用关键词权重"),
    min_df: int = Query(2, description="最小文档频率"),
    max_df: float = Query(0.95, description="最大文档频率")
):
    """上传Excel文件并开始分析"""
    try:
        # 生成唯一任务ID
        job_id = str(uuid.uuid4())
        
        # 保存上传的文件
        file_path = f"uploads/{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 准备分析参数
        params = AnalysisRequest(
            job_id=job_id,
            cluster_method=cluster_method,
            num_clusters=num_clusters,
            threshold=threshold,
            use_weights=use_weights,
            min_df=min_df,
            max_df=max_df
        )
        
        # 初始化任务状态
        analysis_tasks[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "任务已提交，等待处理...",
            "result_file": None
        }
        
        # 启动后台任务
        background_tasks.add_task(analyze_background, file_path, job_id, params)
        
        return AnalysisStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="任务已提交，等待处理..."
        )
    
    except Exception as e:
        logger.error(f"提交任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")

@app.get("/api/status/{job_id}", response_model=AnalysisStatus)
async def get_status(job_id: str):
    """获取分析任务状态"""
    if job_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = analysis_tasks[job_id]
    return AnalysisStatus(
        job_id=job_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        result_file=task["result_file"]
    )

@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """下载分析结果"""
    if job_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = analysis_tasks[job_id]
    if task["status"] != "completed" or not task["result_file"]:
        raise HTTPException(status_code=400, detail="结果尚未生成完成")
    
    file_path = task["result_file"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    return FileResponse(
        path=file_path, 
        filename=f"关键词分析结果_{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/")
async def read_root():
    """API根路径响应"""
    return {"message": "Amazon关键词智能分析系统API", "version": "3.5"}

# Railway 部署所需的配置
if __name__ == "__main__":
    # 获取端口，默认为8000
    port = int(os.environ.get("PORT", 8000))
    # 启动服务
    uvicorn.run("main:app", host="0.0.0.0", port=port)
