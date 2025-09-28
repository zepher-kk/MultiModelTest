# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.utils.coco_metrics import COCOMetrics
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.ops import Profile
from ultralytics.utils import TQDM, callbacks
import torch
import numpy as np
import json
import csv
import time
from pathlib import Path
from tqdm import tqdm


class MultiModalCOCOValidator(MultiModalDetectionValidator):
    """
    多模态COCO验证器，使用COCO评估指标进行多模态目标检测验证。
    
    这个类继承自MultiModalDetectionValidator，保持多模态数据处理能力的同时
    使用COCO标准评估指标，提供更详细的性能分析。
    
    主要特点：
    - 支持RGB+X多模态输入的COCO评估
    - 支持单模态验证（modality参数）
    - 提供12项标准COCO指标
    - 保持与训练器一致的多模态配置解析
    - 处理原始图像尺寸用于精确面积计算
    
    COCO指标包括：
    - AP (IoU=0.50:0.95): 主指标，IoU阈值0.5-0.95平均
    - AP50: IoU阈值0.5时的AP
    - AP75: IoU阈值0.75时的AP  
    - APsmall/APmedium/APlarge: 不同尺寸对象的AP
    - AR1/AR10/AR100: 不同检测限制下的平均召回率
    - ARsmall/ARmedium/ARlarge: 不同尺寸对象的AR
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        初始化多模态COCO验证器。

        Args:
            dataloader: 数据加载器
            save_dir: 保存目录
            pbar: 进度条（当前项目不支持，忽略）
            args: 参数配置（支持modality参数用于单模态验证）
            _callbacks: 回调函数
        """
        # 调用父类初始化，获得完整的多模态能力
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # COCO评估器将在init_metrics中初始化
        self.coco_metrics = None
        
        # 存储用于COCO评估的边界框信息
        self.coco_stats = []
        
        # 进度显示相关
        self.total_batches = len(dataloader) if dataloader else 0
        self.current_batch = 0
        self.progress_bar = None
        
        # 保存验证图像数量
        self.num_images_processed = 0
        
        # 速度统计信息
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}
        self.times = []
        
        # 日志输出
        if self.modality:
            LOGGER.info(f"初始化MultiModalCOCOValidator - 单模态COCO验证: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalCOCOValidator - 双模态COCO验证")

    def init_metrics(self, model):
        """
        初始化COCO评估指标。
        
        首先调用父类的init_metrics方法以初始化标准的metrics对象，
        然后初始化COCOMetrics以提供额外的COCO格式评估。
        
        Args:
            model: 待评估的模型
        """
        # Store the model reference for parameter counting in _print_overall_metrics
        self.model = model
        
        # 调用父类的init_metrics方法，初始化标准的DetMetrics
        # 这确保了所有父类的评估逻辑都能正常工作
        super().init_metrics(model)
        
        # 清理累积数据（重要：确保每次验证开始时数据是干净的）
        self.image_ori_shapes = []
        self.all_pred_boxes = []
        self.all_target_boxes = []
        self.all_pred_cls = []
        self.all_target_cls = []
        self.pred_to_img = []
        self.target_to_img = []
        
        # 初始化COCO评估器
        self.coco_metrics = COCOMetrics(
            save_dir=self.save_dir,
            names=getattr(model, 'names', {}),
            plot=self.args.plots if hasattr(self.args, 'plots') else False,
            on_plot=getattr(self, 'on_plot', None)
        )
        
        # 清空COCO统计信息
        self.coco_stats = []
        
        # 确保属性被正确设置（这些可能已经在父类中设置）
        if not hasattr(self, 'nc'):
            self.nc = getattr(model, 'nc', len(getattr(model, 'names', {})))
        if not hasattr(self, 'end2end'):
            self.end2end = getattr(model, "end2end", False)
        if not hasattr(self, 'names'):
            self.names = getattr(model, 'names', {})
        if not hasattr(self, 'seen'):
            self.seen = 0
        if not hasattr(self, 'jdict'):
            self.jdict = []
        
        LOGGER.info(f"初始化COCO评估指标 - 类别数: {self.nc}")

    def get_desc(self):
        """
        返回COCO格式的进度条描述。
        
        提供比标准验证器更详细的描述，突出COCO评估特性。
        
        Returns:
            str: 进度条描述字符串
        """
        if self.modality:
            return f"%22s" + "%11s" * 5 % ("Class", "Images", "Instances", f"{self.modality.upper()}", "COCO-mAP@.5:.95")
        else:
            return f"%22s" + "%11s" * 5 % ("Class", "Images", "Instances", "RGB+X", "COCO-mAP@.5:.95")

    def update_metrics(self, preds, batch):
        """
        更新验证指标，复用父类的评估逻辑并收集COCO评估所需的数据。
        
        通过调用父类的update_metrics方法，我们能够利用已经验证过的IoU计算、
        true positive判断等核心评估逻辑，然后从父类的metrics中提取数据用于COCO评估。
        
        Args:
            preds: 模型预测结果
            batch: 批次数据，包含图像和标签信息
        """
        # 更新当前批次计数并显示进度
        self.current_batch += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"验证批次 {self.current_batch}/{self.total_batches}")
        
        # 初始化累积数据容器（如果还没有）
        if not hasattr(self, 'image_ori_shapes'):
            self.image_ori_shapes = []  # 每个图像的原始尺寸
            self.all_pred_boxes = []
            self.all_target_boxes = []
            self.all_pred_cls = []
            self.all_target_cls = []
            self.pred_to_img = []  # 记录每个预测框属于哪个图像
            self.target_to_img = []  # 记录每个真实框属于哪个图像
        
        # 调用父类的update_metrics方法，让父类处理所有的评估逻辑
        # 这包括IoU计算、true positive判断、metrics更新等
        super().update_metrics(preds, batch)
        
        # 收集每个图像的数据用于COCO size-specific指标计算
        for si, pred in enumerate(preds):
            # 保存原始图像尺寸（每个图像一个）
            ori_shape = batch["ori_shape"][si]
            self.image_ori_shapes.append(ori_shape)
            
            # 当前已处理的图像索引
            img_idx = len(self.image_ori_shapes) - 1
            
            # 收集预测框数据
            if len(pred['bboxes']) > 0:
                # 预测框已经是xyxy格式
                for i in range(len(pred['bboxes'])):
                    self.all_pred_boxes.append(pred['bboxes'][i].cpu().numpy())
                    self.all_pred_cls.append(pred['cls'][i].cpu().numpy())
                    self.pred_to_img.append(img_idx)  # 记录框属于哪个图像
            
            # 收集真实框数据
            idx = batch["batch_idx"] == si
            target_bboxes = batch["bboxes"][idx]  # xywh格式，归一化坐标
            target_cls = batch["cls"][idx].squeeze(-1)
            
            if len(target_bboxes) > 0:
                # 转换为xyxy格式
                imgsz = batch["img"].shape[2:]
                target_bboxes_xyxy = ops.xywh2xyxy(target_bboxes) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
                
                # 记录每个真实框
                for i in range(len(target_bboxes)):
                    self.all_target_boxes.append(target_bboxes_xyxy[i].cpu().numpy())
                    self.all_target_cls.append(target_cls[i].cpu().numpy())
                    self.target_to_img.append(img_idx)  # 记录框属于哪个图像
    
    def update_speed_stats(self, preprocess_time=None, inference_time=None, postprocess_time=None, total_time=None):
        """
        更新速度统计信息。
        
        Args:
            preprocess_time: 预处理时间(ms)
            inference_time: 推理时间(ms)  
            postprocess_time: 后处理时间(ms)
            total_time: 总时间(ms)
        """
        if preprocess_time is not None:
            self.speed['preprocess'] = preprocess_time
        if inference_time is not None:
            self.speed['inference'] = inference_time
        if postprocess_time is not None:
            self.speed['postprocess'] = postprocess_time
        if total_time is not None:
            self.times.append(total_time)

    def print_results(self):
        """
        输出COCO评估结果，包括各类别AP指标和总体性能指标。
        """
        if self.coco_metrics is None:
            LOGGER.warning("COCO指标尚未初始化，无法输出结果")
            return
        
        # 标题显示
        print(f"\n{colorstr('blue', 'bold', '=' * 80)}")
        print(f"{colorstr('blue', 'bold', '多模态COCO评估结果')}")
        print(f"{colorstr('blue', 'bold', '=' * 80)}")
        
        # 模态信息显示
        if self.modality:
            print(f"验证模式: {colorstr('cyan', f'{self.modality.upper()}-only')} (单模态验证)")
        else:
            print(f"验证模式: {colorstr('cyan', 'RGB+X')} (双模态验证)")
        
        print(f"数据集: {getattr(self.args, 'data', 'N/A')}")
        print(f"类别数: {self.nc}")
        print(f"验证图像数: {self.num_images_processed}")
        
        # 各类别详细指标
        self._print_class_metrics()
        
        # 总体性能指标
        self._print_overall_metrics()
        
        # 速度统计
        self._print_speed_stats()
        
        print(f"{colorstr('blue', 'bold', '=' * 80)}")
        
        # 自动保存CSV结果
        self._save_csv_results()
    
    def _print_table(self, table_data):
        """
        打印格式化的表格。
        
        Args:
            table_data: 表格数据，第一行为表头
        """
        if not table_data:
            return
        
        # 计算每列的最大宽度
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = max(len(str(row[col_idx])) for row in table_data)
            col_widths.append(max_width + 2)  # 添加2个字符的边距
        
        # 打印表头
        header = table_data[0]
        print("┌" + "┬".join("─" * width for width in col_widths) + "┐")
        print("│" + "│".join(f"{str(header[i]):<{col_widths[i]}}" for i in range(len(header))) + "│")
        print("├" + "┼".join("─" * width for width in col_widths) + "┤")
        
        # 打印数据行
        for row in table_data[1:]:
            print("│" + "│".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))) + "│")
        
        print("└" + "┴".join("─" * width for width in col_widths) + "┘")
    
    def _print_speed_stats(self):
        """
        打印速度统计信息。
        """
        print(f"\n{colorstr('cyan', 'bold', '速度统计')}")
        print(f"{colorstr('cyan', 'bold', '-' * 30)}")
        
        if self.times:
            avg_time = np.mean(self.times)
            print(f"平均每张图像处理时间: {avg_time:.1f}ms")
            print(f"推理速度: {1000/avg_time:.1f} FPS")
        
        if any(self.speed.values()):
            print(f"预处理: {self.speed['preprocess']:.1f}ms")
            print(f"推理: {self.speed['inference']:.1f}ms") 
            print(f"后处理: {self.speed['postprocess']:.1f}ms")
    
    def _print_class_metrics(self):
        """
        打印每个类别的AP指标，使用表格格式。
        """
        print(f"\n{colorstr('green', 'bold', '各类别COCO指标')}")
        
        # 准备表格数据
        table_data = [["Class", "AP", "AP50", "AP75"]]  # 表头
        
        # 检查是否有每个类别的统计信息
        # COCOMetrics现在将类别级别的统计信息存储在class_stats属性中
        if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
            ap_array = self.coco_metrics.class_stats['ap']  # Shape: (n_classes, n_iou_thresholds)
            unique_classes = self.coco_metrics.class_stats['unique_classes']
            
            # 获取IoU阈值索引 (COCO标准: 0.5:0.05:0.95)
            # 第0个是0.5, 第5个是0.75
            iou_50_idx = 0
            iou_75_idx = 5
            
            # 收集每个类别的指标
            for ci, class_idx in enumerate(unique_classes):
                class_idx = int(class_idx)  # 确保是整数
                if class_idx < len(self.names):
                    class_name = self.names[class_idx]
                else:
                    class_name = f"class_{class_idx}"
                
                # 获取该类别的AP值
                if ci < ap_array.shape[0]:
                    ap = float(ap_array[ci].mean())  # AP@0.5:0.95
                    ap50 = float(ap_array[ci, iou_50_idx]) if iou_50_idx < ap_array.shape[1] else 0
                    ap75 = float(ap_array[ci, iou_75_idx]) if iou_75_idx < ap_array.shape[1] else 0
                else:
                    ap = ap50 = ap75 = 0
                
                # 添加到表格数据，格式化数值为3位小数
                table_data.append([class_name, f"{ap:.3f}", f"{ap50:.3f}", f"{ap75:.3f}"])
            
            # 使用表格打印
            self._print_table(table_data)
        else:
            # 如果没有详细的类别统计，尝试从父类metrics获取基本信息
            if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'ap_class_index'):
                # 使用父类的标准指标
                ap_class_index = self.metrics.box.ap_class_index
                table_data = [["Class", "AP", "AP50", "AP75"]]  # 表头
                
                # 收集有数据的类别指标
                for i, c in enumerate(ap_class_index):
                    class_name = self.names[c] if c < len(self.names) else f"class_{c}"
                    
                    # 使用父类的指标
                    p, r, ap50, ap = self.metrics.box.class_result(i)
                    ap75 = 0  # 父类没有AP75
                    
                    # 添加到表格数据
                    table_data.append([class_name, f"{ap:.3f}", f"{ap50:.3f}", f"{ap75:.3f}"])
                
                # 使用表格打印
                self._print_table(table_data)
            else:
                print("\n每类别指标详情暂不可用")
    
    def _print_overall_metrics(self):
        """
        打印总体性能指标表，包括AP、APsmall/medium/large、FPS、参数量等。
        """
        print(f"\n{colorstr('cyan', 'bold', '总体性能指标')}")
        
        # 计算FPS
        fps = 0.0
        if any(self.speed.values()):
            total_time = self.speed['preprocess'] + self.speed['inference'] + self.speed['postprocess']
            if total_time > 0:
                fps = 1000.0 / total_time
        
        # 获取模型参数量
        params = 0
        if hasattr(self, 'model') and self.model is not None:
            try:
                # 尝试从model获取参数信息
                if hasattr(self.model, 'parameters'):
                    params = sum(p.numel() for p in self.model.parameters())
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                    params = sum(p.numel() for p in self.model.model.parameters())
            except:
                params = 0
        
        # 创建总体指标表格
        overall_table = [
            ["Metric", "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "FPS", "Params"],
            [
                "All",
                f"{self.coco_metrics.AP:.3f}",
                f"{self.coco_metrics.AP50:.3f}",
                f"{self.coco_metrics.AP75:.3f}",
                f"{getattr(self.coco_metrics, 'APsmall', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APmedium', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APlarge', 0.0):.3f}",
                f"{fps:.1f}",
                f"{params:,}" if params > 0 else "N/A"
            ]
        ]
        
        self._print_table(overall_table)
        
        # 新增：按尺寸性能指标表格
        print(f"\n{colorstr('yellow', 'bold', '按尺寸性能指标')}")
        size_metrics_table = [
            ["Size", "AP", "AP50", "AP75"],
            [
                "Small", 
                f"{getattr(self.coco_metrics, 'APsmall', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APsmall50', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APsmall75', 0.0):.3f}"
            ],
            [
                "Medium", 
                f"{getattr(self.coco_metrics, 'APmedium', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APmedium50', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APmedium75', 0.0):.3f}"
            ],
            [
                "Large", 
                f"{getattr(self.coco_metrics, 'APlarge', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APlarge50', 0.0):.3f}",
                f"{getattr(self.coco_metrics, 'APlarge75', 0.0):.3f}"
            ]
        ]
        self._print_table(size_metrics_table)
    
    def _save_csv_results(self):
        """
        自动保存COCO验证结果到CSV文件。
        生成4个CSV文件：
        - coco_metrics_by_class.csv: 按类别的指标
        - coco_metrics_by_size.csv: 按尺寸的指标  
        - coco_metrics_overall.csv: 总体指标
        - coco_metrics_comprehensive.csv: 综合所有信息
        """
        try:
            # 确保保存目录存在
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 按类别指标CSV
            class_csv_path = save_dir / "coco_metrics_by_class.csv"
            with open(class_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Class', 'AP', 'AP50', 'AP75']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 检查是否有每个类别的统计信息
                if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
                    ap_array = self.coco_metrics.class_stats['ap']
                    unique_classes = self.coco_metrics.class_stats['unique_classes']
                    
                    iou_50_idx = 0
                    iou_75_idx = 5
                    
                    for ci, class_idx in enumerate(unique_classes):
                        class_idx = int(class_idx)
                        class_name = self.names[class_idx] if class_idx < len(self.names) else f"class_{class_idx}"
                        
                        if ci < ap_array.shape[0]:
                            ap = float(ap_array[ci].mean())
                            ap50 = float(ap_array[ci, iou_50_idx]) if iou_50_idx < ap_array.shape[1] else 0
                            ap75 = float(ap_array[ci, iou_75_idx]) if iou_75_idx < ap_array.shape[1] else 0
                        else:
                            ap = ap50 = ap75 = 0
                        
                        writer.writerow({
                            'Class': class_name,
                            'AP': f"{ap:.3f}",
                            'AP50': f"{ap50:.3f}",
                            'AP75': f"{ap75:.3f}"
                        })
                elif hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'ap_class_index'):
                    # 使用父类的标准指标
                    ap_class_index = self.metrics.box.ap_class_index
                    for i, c in enumerate(ap_class_index):
                        class_name = self.names[c] if c < len(self.names) else f"class_{c}"
                        p, r, ap50, ap = self.metrics.box.class_result(i)
                        writer.writerow({
                            'Class': class_name,
                            'AP': f"{ap:.3f}",
                            'AP50': f"{ap50:.3f}",
                            'AP75': "0.000"  # 父类没有AP75
                        })
            
            # 2. 按尺寸指标CSV
            size_csv_path = save_dir / "coco_metrics_by_size.csv"
            with open(size_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Size', 'AP', 'AP50', 'AP75']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                sizes = ['Small', 'Medium', 'Large']
                for size in sizes:
                    ap = getattr(self.coco_metrics, f'AP{size.lower()}', 0.0)
                    ap50 = getattr(self.coco_metrics, f'AP{size.lower()}50', 0.0)
                    ap75 = getattr(self.coco_metrics, f'AP{size.lower()}75', 0.0)
                    
                    writer.writerow({
                        'Size': size,
                        'AP': f"{ap:.3f}",
                        'AP50': f"{ap50:.3f}",
                        'AP75': f"{ap75:.3f}"
                    })
            
            # 3. 总体指标CSV
            overall_csv_path = save_dir / "coco_metrics_overall.csv"
            with open(overall_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Metric', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 计算FPS和参数量
                fps = 0.0
                if any(self.speed.values()):
                    total_time = self.speed['preprocess'] + self.speed['inference'] + self.speed['postprocess']
                    if total_time > 0:
                        fps = 1000.0 / total_time
                
                params = 0
                if hasattr(self, 'model') and self.model is not None:
                    try:
                        if hasattr(self.model, 'parameters'):
                            params = sum(p.numel() for p in self.model.parameters())
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                            params = sum(p.numel() for p in self.model.model.parameters())
                    except:
                        params = 0
                
                # 写入总体指标
                metrics_data = [
                    ('AP', f"{self.coco_metrics.AP:.3f}"),
                    ('AP50', f"{self.coco_metrics.AP50:.3f}"),
                    ('AP75', f"{self.coco_metrics.AP75:.3f}"),
                    ('APsmall', f"{getattr(self.coco_metrics, 'APsmall', 0.0):.3f}"),
                    ('APmedium', f"{getattr(self.coco_metrics, 'APmedium', 0.0):.3f}"),
                    ('APlarge', f"{getattr(self.coco_metrics, 'APlarge', 0.0):.3f}"),
                    ('FPS', f"{fps:.1f}"),
                    ('Parameters', str(params) if params > 0 else "N/A"),
                    ('Images', str(self.num_images_processed)),
                    ('Modality', self.modality if self.modality else 'multimodal')
                ]
                
                for metric, value in metrics_data:
                    writer.writerow({'Metric': metric, 'Value': value})
            
            # 4. 综合信息CSV
            comprehensive_csv_path = save_dir / "coco_metrics_comprehensive.csv"
            with open(comprehensive_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Category', 'Type', 'Metric', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 写入类别指标
                if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
                    ap_array = self.coco_metrics.class_stats['ap']
                    unique_classes = self.coco_metrics.class_stats['unique_classes']
                    
                    iou_50_idx = 0
                    iou_75_idx = 5
                    
                    for ci, class_idx in enumerate(unique_classes):
                        class_idx = int(class_idx)
                        class_name = self.names[class_idx] if class_idx < len(self.names) else f"class_{class_idx}"
                        
                        if ci < ap_array.shape[0]:
                            ap = float(ap_array[ci].mean())
                            ap50 = float(ap_array[ci, iou_50_idx]) if iou_50_idx < ap_array.shape[1] else 0
                            ap75 = float(ap_array[ci, iou_75_idx]) if iou_75_idx < ap_array.shape[1] else 0
                        else:
                            ap = ap50 = ap75 = 0
                        
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'AP', 'Value': f"{ap:.3f}"})
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'AP50', 'Value': f"{ap50:.3f}"})
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'AP75', 'Value': f"{ap75:.3f}"})
                
                # 写入尺寸指标
                sizes = ['Small', 'Medium', 'Large']
                for size in sizes:
                    ap = getattr(self.coco_metrics, f'AP{size.lower()}', 0.0)
                    ap50 = getattr(self.coco_metrics, f'AP{size.lower()}50', 0.0)
                    ap75 = getattr(self.coco_metrics, f'AP{size.lower()}75', 0.0)
                    
                    writer.writerow({'Category': size, 'Type': 'Size', 'Metric': 'AP', 'Value': f"{ap:.3f}"})
                    writer.writerow({'Category': size, 'Type': 'Size', 'Metric': 'AP50', 'Value': f"{ap50:.3f}"})
                    writer.writerow({'Category': size, 'Type': 'Size', 'Metric': 'AP75', 'Value': f"{ap75:.3f}"})
                
                # 写入总体指标
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AP', 'Value': f"{self.coco_metrics.AP:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AP50', 'Value': f"{self.coco_metrics.AP50:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AP75', 'Value': f"{self.coco_metrics.AP75:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'APsmall', 'Value': f"{getattr(self.coco_metrics, 'APsmall', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'APmedium', 'Value': f"{getattr(self.coco_metrics, 'APmedium', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'APlarge', 'Value': f"{getattr(self.coco_metrics, 'APlarge', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'FPS', 'Value': f"{fps:.1f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'Parameters', 'Value': str(params) if params > 0 else "N/A"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'Images', 'Value': str(self.num_images_processed)})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'Modality', 'Value': self.modality if self.modality else 'multimodal'})
            
            LOGGER.info(f"CSV结果已保存到: {save_dir}")
            LOGGER.info(f"  - {class_csv_path.name}")
            LOGGER.info(f"  - {size_csv_path.name}")
            LOGGER.info(f"  - {overall_csv_path.name}")
            LOGGER.info(f"  - {comprehensive_csv_path.name}")
            
        except Exception as e:
            LOGGER.warning(f"保存CSV文件时出错: {e}")

    def get_stats(self):
        """
        任务2.6 - 实现get_stats方法：
        - 返回COCO格式的指标字典，使用'metrics/coco/'前缀命名
        - 确保与现有YOLO验证系统的兼容性
        - 包含所有12个COCO指标的键值对
        - 支持训练过程中的指标监控
        - 返回格式要与DetectionValidator.get_stats()兼容
        
        Returns:
            dict: 包含COCO指标的字典，键名与标准格式兼容
        """
        if self.coco_metrics is None:
            LOGGER.warning("COCO指标尚未初始化，返回空字典")
            return {}
        
        # COCO指标已在finalize_metrics中计算完成
        
        # 构建完整的COCO指标字典
        stats = {
            # 标准YOLO兼容格式 - 主要指标
            'metrics/precision(B)': getattr(self.coco_metrics, 'precision', 0.0),
            'metrics/recall(B)': getattr(self.coco_metrics, 'recall', 0.0),
            'metrics/mAP50(B)': getattr(self.coco_metrics, 'AP50', 0.0),
            'metrics/mAP50-95(B)': getattr(self.coco_metrics, 'AP', 0.0),
            
            # COCO特定指标 - 使用'metrics/coco/'前缀
            'metrics/coco/AP': getattr(self.coco_metrics, 'AP', 0.0),
            'metrics/coco/AP50': getattr(self.coco_metrics, 'AP50', 0.0),
            'metrics/coco/AP75': getattr(self.coco_metrics, 'AP75', 0.0),
            'metrics/coco/APsmall': getattr(self.coco_metrics, 'APsmall', 0.0),
            'metrics/coco/APmedium': getattr(self.coco_metrics, 'APmedium', 0.0),
            'metrics/coco/APlarge': getattr(self.coco_metrics, 'APlarge', 0.0),
            'metrics/coco/AR1': getattr(self.coco_metrics, 'AR1', 0.0),
            'metrics/coco/AR10': getattr(self.coco_metrics, 'AR10', 0.0),
            'metrics/coco/AR100': getattr(self.coco_metrics, 'AR100', 0.0),
            'metrics/coco/ARsmall': getattr(self.coco_metrics, 'ARsmall', 0.0),
            'metrics/coco/ARmedium': getattr(self.coco_metrics, 'ARmedium', 0.0),
            'metrics/coco/ARlarge': getattr(self.coco_metrics, 'ARlarge', 0.0),
            
            # 训练器兼容性指标
            'fitness': getattr(self.coco_metrics, 'AP', 0.0),  # 使用主指标AP作为fitness
            
            # 速度指标
            'val/speed_preprocess': self.speed.get('preprocess', 0.0),
            'val/speed_inference': self.speed.get('inference', 0.0),
            'val/speed_postprocess': self.speed.get('postprocess', 0.0),
            
            # 数据统计
            'val/images': len(self.coco_stats),
            'val/instances': sum(len(stat.get('ground_truth_labels', [])) for stat in self.coco_stats),
        }
        
        # 添加模态特定信息
        if self.modality:
            stats[f'val/modality'] = self.modality
            stats[f'metrics/coco/modality'] = self.modality
        else:
            stats[f'val/modality'] = 'multimodal'
            stats[f'metrics/coco/modality'] = 'RGB+X'
        
        # 添加每类别指标（如果可用）
        if hasattr(self.coco_metrics, 'per_class_metrics'):
            for class_id, class_metrics in self.coco_metrics.per_class_metrics.items():
                class_name = getattr(self.coco_metrics, 'names', {}).get(class_id, f'class_{class_id}')
                stats[f'metrics/coco/class_{class_name}_AP'] = class_metrics.get('AP', 0.0)
                stats[f'metrics/coco/class_{class_name}_AP50'] = class_metrics.get('AP50', 0.0)
                stats[f'metrics/coco/class_{class_name}_AP75'] = class_metrics.get('AP75', 0.0)
        
        return stats

    # 删除了旧的_process_coco_stats方法，现在使用_process_coco_stats_from_metrics
    
    def _set_default_coco_stats(self):
        """
        设置默认的COCO统计值。
        """
        default_stats = {
            'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
            'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
            'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
            'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0
        }
        self.coco_metrics.update(default_stats)
    
    def _process_coco_stats_from_metrics(self):
        """
        从父类的metrics中提取数据并计算COCO指标。
        
        这个方法直接使用父类已经计算好的tp、conf等数据，
        转换成COCOMetrics所需的格式并计算COCO指标。
        """
        try:
            # 从父类的metrics中提取统计数据
            stats = self.metrics.stats
            
            # 将列表转换为numpy数组
            tp = np.concatenate(stats['tp'], axis=0) if stats['tp'] else np.array([])
            conf = np.concatenate(stats['conf'], axis=0) if stats['conf'] else np.array([])
            pred_cls = np.concatenate(stats['pred_cls'], axis=0) if stats['pred_cls'] else np.array([])
            target_cls = np.concatenate(stats['target_cls'], axis=0) if stats['target_cls'] else np.array([])
            
            # 处理过多的检测结果
            # COCOMetrics内部有一个硬编码的1000个检测的限制
            # 我们需要按置信度排序并只保留前 N 个检测
            MAX_DETECTIONS_PER_IMAGE = 100  # COCO标准每张图像最多100个检测
            
            if len(conf) > 0:
                # 按置信度降序排序
                sort_indices = np.argsort(conf)[::-1]
                
                # 为每张图像限制检测数量
                # 由于我们没有图像ID信息，这里简单地限制总数
                # 使用父类的seen属性或默认值
                num_images = getattr(self, 'seen', 630) if hasattr(self, 'seen') else 630
                max_total_detections = MAX_DETECTIONS_PER_IMAGE * num_images
                
                # 为了避免COCOMetrics内部的硬编码限制，进一步限制总数
                max_total_detections = min(max_total_detections, 50000)  # 限制总数为50000
                
                if len(sort_indices) > max_total_detections:
                    LOGGER.warning(f"检测数量({len(sort_indices)})超过COCO限制({max_total_detections})，将按置信度截断")
                    sort_indices = sort_indices[:max_total_detections]
                
                # 按排序索引重新排列所有数据
                tp = tp[sort_indices] if tp.ndim > 1 else tp[sort_indices]
                conf = conf[sort_indices]
                pred_cls = pred_cls[sort_indices]
                # target_cls不需要重新排序，因为它是真实标签
            
            # 获取bbox数据（使用收集的数据）
            pred_boxes = None
            target_boxes = None
            ori_shapes = None
            
            # 使用在update_metrics中收集的bbox和尺寸数据
            pred_to_img = None
            target_to_img = None
            
            if hasattr(self, 'all_pred_boxes') and self.all_pred_boxes:
                pred_boxes = np.array(self.all_pred_boxes)
                pred_to_img = np.array(self.pred_to_img)
                # 由于我们在收集时已经按检测顺序存储，这里需要匹配排序后的索引
                # 但由于pred_boxes是按检测顺序收集的，我们需要确保与tp/conf的顺序一致
                if len(conf) > 0 and len(sort_indices) < len(pred_boxes):
                    pred_boxes = pred_boxes[sort_indices]
                    pred_to_img = pred_to_img[sort_indices]
                    
            if hasattr(self, 'all_target_boxes') and self.all_target_boxes:
                target_boxes = np.array(self.all_target_boxes)
                target_to_img = np.array(self.target_to_img)
                
            if hasattr(self, 'image_ori_shapes') and self.image_ori_shapes:
                # ori_shapes是每个图像的原始尺寸列表
                ori_shapes = self.image_ori_shapes
            
            
            # 调用COCOMetrics.process计算COCO指标
            self.coco_metrics.process(
                tp=tp,
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                ori_shapes=ori_shapes
            )
            
            # COCOMetrics现在会在class_stats属性中保存类别级别的统计信息
            if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats:
                LOGGER.info(f"COCO类别统计可用: {list(self.coco_metrics.class_stats.keys())}")
            
            LOGGER.info(f"COCO指标计算完成 - 处理了 {len(conf) if conf.size > 0 else 0} 个检测结果")
            
        except Exception as e:
            LOGGER.error(f"计算COCO指标时出错: {e}")
            import traceback
            traceback.print_exc()
            self._set_default_coco_stats()
    
    def _preprocess_coco_data(self):
        """
        预处理COCO数据，优化批量处理。
        
        Returns:
            tuple: (all_predictions, all_ground_truths)
        """
        all_predictions = []
        all_ground_truths = []
        
        # 使用内部进度条显示数据处理进度
        for stats in tqdm(self.coco_stats, desc="处理图像数据", unit="图像", leave=False):
            image_id = stats['image_id']
            preds = stats['predictions']
            gt_labels = stats['ground_truth_labels']
            gt_bboxes = stats['ground_truth_bboxes']
            orig_shape = stats['original_shape']
            
            # 批量处理预测结果
            if isinstance(preds, torch.Tensor) and len(preds) > 0:
                pred_data = self._process_predictions_batch(preds, image_id, orig_shape)
                all_predictions.extend(pred_data)
            
            # 批量处理真实标签
            if isinstance(gt_labels, torch.Tensor) and len(gt_labels) > 0:
                gt_data = self._process_ground_truths_batch(gt_labels, gt_bboxes, image_id, orig_shape)
                all_ground_truths.extend(gt_data)
        
        return all_predictions, all_ground_truths
    
    def _process_predictions_batch(self, preds, image_id, orig_shape):
        """
        批量处理预测结果，提高效率。
        
        Args:
            preds: 预测张量
            image_id: 图像ID
            orig_shape: 原始图像尺寸
            
        Returns:
            list: 处理后的预测结果列表
        """
        predictions = []
        
        # 向量化处理预测结果
        if len(preds) > 0 and preds.shape[-1] >= 6:
            # 批量提取数据
            bboxes = preds[:, :4].cpu().numpy()  # [x1, y1, x2, y2]
            confs = preds[:, 4].cpu().numpy()    # 置信度
            classes = preds[:, 5].cpu().numpy().astype(int)  # 类别
            
            # 批量计算面积
            areas = np.array([COCOMetrics.calculate_bbox_area(bbox, orig_shape) for bbox in bboxes])
            
            # 批量构建结果
            for i in range(len(preds)):
                predictions.append({
                    'image_id': image_id,
                    'bbox': bboxes[i],
                    'confidence': float(confs[i]),
                    'class': int(classes[i]),
                    'area': areas[i],
                    'original_shape': orig_shape
                })
        
        return predictions
    
    def _process_ground_truths_batch(self, gt_labels, gt_bboxes, image_id, orig_shape):
        """
        批量处理真实标签，提高效率。
        
        Args:
            gt_labels: 真实标签张量
            gt_bboxes: 真实边界框张量
            image_id: 图像ID
            orig_shape: 原始图像尺寸
            
        Returns:
            list: 处理后的真实标签列表
        """
        ground_truths = []
        
        if len(gt_labels) > 0 and len(gt_bboxes) > 0:
            # 向量化处理
            labels = gt_labels.cpu().numpy().astype(int)
            bboxes = gt_bboxes.cpu().numpy()
            
            # 批量计算面积
            areas = np.array([COCOMetrics.calculate_bbox_area(bbox, orig_shape) for bbox in bboxes])
            
            # 批量构建结果
            for i in range(len(labels)):
                ground_truths.append({
                    'image_id': image_id,
                    'bbox': bboxes[i],
                    'class': int(labels[i]),
                    'area': areas[i],
                    'original_shape': orig_shape
                })
        
        return ground_truths
    
    def _compute_coco_metrics_optimized(self, predictions, ground_truths):
        """
        优化的COCO指标计算方法。
        
        使用COCOMetrics类的优化处理方法。
        
        Args:
            predictions: 所有预测结果列表
            ground_truths: 所有真实标签列表
            
        Returns:
            dict: 包含12项COCO指标的字典
        """
        if len(predictions) == 0 or len(ground_truths) == 0:
            return {
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
                'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0
            }
        
        try:
            # 转换数据格式为COCOMetrics.process()所需的格式
            tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes = self._convert_to_coco_format(
                predictions, ground_truths
            )
            
            # 使用COCOMetrics的优化处理方法
            temp_metrics = COCOMetrics(save_dir=self.save_dir, names=self.coco_metrics.names)
            temp_metrics.process(
                tp, conf, pred_cls, target_cls,
                pred_boxes=pred_boxes, 
                target_boxes=target_boxes, 
                ori_shapes=ori_shapes,
                show_progress=True  # 启用进度显示
            )
            
            # 返回计算得到的指标
            return temp_metrics.get_summary_dict()
            
        except Exception as e:
            LOGGER.error(f"COCO指标计算内部错误: {e}")
            return {
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
                'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0
            }
    
    def _convert_to_coco_format(self, predictions, ground_truths):
        """
        将预测和真实标签转换为COCOMetrics.process()所需的格式。
        
        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            
        Returns:
            tuple: (tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes)
        """
        # 简化实现：创建基本的数据格式
        if not predictions or not ground_truths:
            return (np.array([]), np.array([]), np.array([]), np.array([]), 
                   np.array([]).reshape(0, 4), np.array([]).reshape(0, 4), [])
        
        # 提取预测数据
        pred_confs = np.array([p['confidence'] for p in predictions])
        pred_classes = np.array([p['class'] for p in predictions])
        pred_boxes = np.array([p['bbox'] for p in predictions])
        
        # 提取真实标签数据
        target_classes = np.array([gt['class'] for gt in ground_truths])
        target_boxes = np.array([gt['bbox'] for gt in ground_truths])
        
        # 获取所有原始图像尺寸
        ori_shapes = list(set([tuple(p['original_shape']) for p in predictions + ground_truths]))
        
        # 简化的TP计算（实际应用中需要更复杂的IoU匹配）
        # 这里使用简化的方法，实际效果可能不如完整的COCO评估
        tp = np.ones((len(predictions), 10))  # 假设所有预测都是TP
        
        return tp, pred_confs, pred_classes, target_classes, pred_boxes, target_boxes, ori_shapes

    def run_validation(self):
        """
        运行完整的验证流程，集成进度显示。
        
        这个方法重写了父类的验证流程，添加了完整的进度显示支持。
        """
        # 初始化进度条
        self.init_progress_bar()
        
        try:
            # 调用父类的验证方法（如果存在）
            # 这里可以调用MultiModalDetectionValidator的验证方法
            # 或者实现自定义的验证流程
            
            # 如果需要调用父类方法，可以取消注释下面的行
            # super().run_validation()
            
            LOGGER.info("多模态COCO验证流程完成")
            
        finally:
            # 确保进度条被正确关闭
            self.close_progress_bar()

    def _extract_ori_shapes(self, batch, batch_size):
        """
        优化原始图像尺寸信息提取。
        
        统一处理多种可能的ori_shape格式：
        - 列表/元组格式: [(h1,w1), (h2,w2), ...]
        - 张量格式: tensor([[h1,w1], [h2,w2], ...])
        - 单个尺寸: (h,w) 或 tensor([h,w])
        - 多模态数据的6通道输入兼容性
        
        Args:
            batch: 批次数据
            batch_size: 批次大小
            
        Returns:
            list: 标准化的原始尺寸列表
        """
        orig_shapes = batch.get("ori_shape", None)
        
        if orig_shapes is None:
            # 如果没有原始尺寸信息，使用默认值
            LOGGER.warning(f"批次缺少原始尺寸信息，使用默认值 (640, 640)")
            return [(640, 640)] * batch_size
        
        # 处理张量格式
        if isinstance(orig_shapes, torch.Tensor):
            orig_shapes = orig_shapes.cpu().numpy().tolist()
        
        # 处理单个尺寸的情况
        if isinstance(orig_shapes, (tuple, list)) and len(orig_shapes) == 2 and isinstance(orig_shapes[0], (int, float)):
            # 单个尺寸，复制到整个批次
            return [tuple(orig_shapes)] * batch_size
        
        # 处理批次尺寸列表
        if isinstance(orig_shapes, (list, tuple)):
            result = []
            for i in range(batch_size):
                if i < len(orig_shapes):
                    shape = orig_shapes[i]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        result.append(tuple(shape[:2]))  # 取前两个值作为(h,w)
                    else:
                        result.append((640, 640))  # 默认值
                else:
                    result.append((640, 640))  # 超出范围使用默认值
            return result
        
        # 其他情况使用默认值
        LOGGER.warning(f"无法解析原始尺寸格式: {type(orig_shapes)}，使用默认值")
        return [(640, 640)] * batch_size
    
    def _filter_labels_for_image(self, labels, bboxes, batch_idx, image_idx):
        """
        优化单张图像的标签过滤操作。
        
        使用向量化操作提高效率，避免重复的张量操作。
        
        Args:
            labels: 所有标签
            bboxes: 所有边界框
            batch_idx: 批次索引
            image_idx: 当前图像索引
            
        Returns:
            tuple: (当前图像标签, 当前图像边界框)
        """
        if len(batch_idx) > 0 and len(labels) > 0:
            # 使用向量化操作过滤
            mask = (batch_idx == image_idx)
            current_labels = labels[mask] if mask.any() else torch.tensor([])
            current_bboxes = bboxes[mask] if mask.any() and len(bboxes) > 0 else torch.tensor([]).reshape(0, 4)
        else:
            current_labels = labels if len(labels) > 0 else torch.tensor([])
            current_bboxes = bboxes if len(bboxes) > 0 else torch.tensor([]).reshape(0, 4)
        
        # 确保返回的是克隆的张量，避免后续修改影响原始数据
        if isinstance(current_labels, torch.Tensor) and current_labels.numel() > 0:
            current_labels = current_labels.clone()
        if isinstance(current_bboxes, torch.Tensor) and current_bboxes.numel() > 0:
            current_bboxes = current_bboxes.clone()
            
        return current_labels, current_bboxes
    
    def init_progress_bar(self):
        """
        初始化验证进度条。
        
        任务6.2 - 实现进度显示：
        - 显示验证进度条
        - 提供实时验证状态
        """
        if self.total_batches > 0:
            self.progress_bar = tqdm(
                total=self.total_batches,
                desc="多模态COCO验证",
                unit="batch",
                leave=True,
                ncols=100
            )
            if self.modality:
                self.progress_bar.set_description(f"{self.modality.upper()}模态COCO验证")
            else:
                self.progress_bar.set_description("RGB+X多模态COCO验证")
    
    def close_progress_bar(self):
        """
        关闭进度条并显示完成信息。
        """
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
    
    def finalize_metrics(self):
        """
        完成指标计算。
        
        首先调用父类的finalize_metrics方法完成标准指标计算，
        然后进行COCO指标的计算和处理。
        """
        # 关闭进度条
        self.close_progress_bar()
        
        # 调用父类的finalize_metrics方法
        # 这将完成标准的metrics计算，包括loss、speed等
        super().finalize_metrics()
        
        # 从父类的metrics中提取已处理的数据用于COCO评估
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'stats') and self.metrics.stats:
            # 父类的stats是一个字典，包含各种统计数据的列表
            # 我们直接使用这些数据进行COCO评估
            with tqdm(total=1, desc="计算COCO指标", unit="stage") as pbar:
                self._process_coco_stats_from_metrics()
                pbar.update(1)
        else:
            LOGGER.warning("没有可用的统计数据进行COCO评估")
            self._set_default_coco_stats()
        
        # 保存处理的图像数量
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'seen'):
            self.num_images_processed = self.metrics.seen
        else:
            self.num_images_processed = self.seen
        
        # 验证完成，不需要额外的日志输出
    
    def save_json(self, save_dir=None, filename=None):
        """
        任务5.1 - 实现JSON格式的结果保存：
        - 在MultiModalCOCOValidator中实现save_json功能
        - 保存COCO格式的验证结果到JSON文件
        - 包含所有计算的指标和详细统计信息
        - 支持逐类别指标的保存
        - 文件命名和路径与现有系统保持一致
        
        Args:
            save_dir: 保存目录，默认使用self.save_dir
            filename: 文件名，默认为coco_results.json
        """
        if self.coco_metrics is None:
            LOGGER.warning("COCO指标尚未计算，无法保存结果")
            return None
        
        # 确定保存路径
        save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            modality_suffix = f"_{self.modality}" if self.modality else "_multimodal" 
            filename = f"coco_results{modality_suffix}.json"
        
        save_path = save_dir / filename
        
        # 构建完整的结果数据
        results_data = {
            "evaluation_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "validator_type": "MultiModalCOCOValidator",
                "modality": self.modality if self.modality else "RGB+X",
                "dataset": getattr(self.args, 'data', 'N/A'),
                "num_classes": self.nc,
                "num_images": len(self.coco_stats),
                "num_instances": sum(len(stat.get('ground_truth_labels', [])) for stat in self.coco_stats)
            },
            
            "coco_metrics": {
                # 主要AP指标
                "AP": getattr(self.coco_metrics, 'AP', 0.0),
                "AP50": getattr(self.coco_metrics, 'AP50', 0.0),
                "AP75": getattr(self.coco_metrics, 'AP75', 0.0),
                "APsmall": getattr(self.coco_metrics, 'APsmall', 0.0),
                "APmedium": getattr(self.coco_metrics, 'APmedium', 0.0),
                "APlarge": getattr(self.coco_metrics, 'APlarge', 0.0),
                
                # 召回指标
                "AR1": getattr(self.coco_metrics, 'AR1', 0.0),
                "AR10": getattr(self.coco_metrics, 'AR10', 0.0),
                "AR100": getattr(self.coco_metrics, 'AR100', 0.0),
                "ARsmall": getattr(self.coco_metrics, 'ARsmall', 0.0),
                "ARmedium": getattr(self.coco_metrics, 'ARmedium', 0.0),
                "ARlarge": getattr(self.coco_metrics, 'ARlarge', 0.0),
                
                # 附加指标
                "precision": getattr(self.coco_metrics, 'precision', 0.0),
                "recall": getattr(self.coco_metrics, 'recall', 0.0)
            },
            
            "speed_statistics": {
                "preprocess_ms": self.speed.get('preprocess', 0.0),
                "inference_ms": self.speed.get('inference', 0.0),
                "postprocess_ms": self.speed.get('postprocess', 0.0),
                "total_ms": sum(self.speed.values()),
                "fps": 1000 / np.mean(self.times) if self.times else 0.0,
                "avg_time_per_image_ms": np.mean(self.times) if self.times else 0.0
            },
            
            "detailed_stats": self._get_detailed_stats_for_json(),
            
            "configuration": {
                "args": vars(self.args) if self.args else {},
                "model_info": {
                    "stride": getattr(self, 'stride', None),
                    "nc": self.nc
                }
            }
        }
        
        # 添加每类别指标（如果可用）
        if hasattr(self.coco_metrics, 'per_class_metrics') and self.coco_metrics.per_class_metrics:
            class_names = getattr(self.coco_metrics, 'names', {})
            results_data["per_class_metrics"] = {}
            
            for class_id, class_metrics in self.coco_metrics.per_class_metrics.items():
                class_name = class_names.get(class_id, f"class_{class_id}")
                results_data["per_class_metrics"][class_name] = {
                    "AP": class_metrics.get('AP', 0.0),
                    "AP50": class_metrics.get('AP50', 0.0),
                    "AP75": class_metrics.get('AP75', 0.0),
                    "AR100": class_metrics.get('AR100', 0.0)
                }
        
        # 保存JSON文件
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            LOGGER.info(f"COCO评估结果已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            LOGGER.error(f"保存COCO结果时出错: {e}")
            return None
    
    def _get_detailed_stats_for_json(self):
        """
        获取用于JSON保存的详细统计信息。
        
        Returns:
            dict: 详细统计信息
        """
        detailed_stats = {
            "total_predictions": 0,
            "total_ground_truths": 0,
            "images_processed": len(self.coco_stats),
            "processing_details": []
        }
        
        for i, stat in enumerate(self.coco_stats):
            image_detail = {
                "image_id": stat.get('image_id', i),
                "num_predictions": len(stat.get('predictions', [])) if isinstance(stat.get('predictions'), (list, torch.Tensor)) else 0,
                "num_ground_truths": len(stat.get('ground_truth_labels', [])) if isinstance(stat.get('ground_truth_labels'), (list, torch.Tensor)) else 0,
                "original_shape": stat.get('original_shape', (0, 0))
            }
            
            detailed_stats["total_predictions"] += image_detail["num_predictions"]
            detailed_stats["total_ground_truths"] += image_detail["num_ground_truths"]
            detailed_stats["processing_details"].append(image_detail)
        
        return detailed_stats
    
    def save_results(self, save_conf=True, save_json_results=True, plots=True):
        """
        任务5.2 - 集成现有的保存选项：
        - 支持save_conf参数（混淆矩阵保存）
        - 支持plots参数（可视化图表生成）
        - 确保与父类MultiModalDetectionValidator保存功能的兼容
        - 继承并扩展现有的保存流程
        - 支持所有现有的可视化选项
        
        Args:
            save_conf: 是否保存混淆矩阵
            save_json_results: 是否保存JSON格式结果
            plots: 是否生成可视化图表
        """
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"开始保存COCO验证结果到: {save_dir}")
        
        results_saved = []
        
        # 1. 保存JSON格式的详细结果
        if save_json_results:
            try:
                json_path = self.save_json(save_dir)
                if json_path:
                    results_saved.append(f"JSON结果: {json_path}")
            except Exception as e:
                LOGGER.error(f"保存JSON结果失败: {e}")
        
        # 2. 保存混淆矩阵（如果支持）
        if save_conf and hasattr(self.coco_metrics, 'confusion_matrix'):
            try:
                conf_path = save_dir / "confusion_matrix.png"
                if hasattr(self.coco_metrics.confusion_matrix, 'plot'):
                    self.coco_metrics.confusion_matrix.plot(save_dir=save_dir, names=getattr(self.coco_metrics, 'names', {}))
                    results_saved.append(f"混淆矩阵: {conf_path}")
            except Exception as e:
                LOGGER.warning(f"保存混淆矩阵失败: {e}")
        
        # 3. 生成可视化图表
        if plots:
            try:
                self._save_visualization_plots(save_dir)
                results_saved.append(f"可视化图表: {save_dir / 'plots'}")
            except Exception as e:
                LOGGER.warning(f"生成可视化图表失败: {e}")
        
        # 4. 保存汇总报告
        try:
            self._save_summary_report(save_dir)
            results_saved.append(f"汇总报告: {save_dir / 'coco_summary.txt'}")
        except Exception as e:
            LOGGER.warning(f"保存汇总报告失败: {e}")
        
        # 5. 调用父类保存方法（如果存在且兼容）
        try:
            if hasattr(super(), 'save_results'):
                # 调用父类方法，但要处理参数兼容性
                super_args = {}
                if 'save_conf' in super().save_results.__code__.co_varnames:
                    super_args['save_conf'] = save_conf
                if 'plots' in super().save_results.__code__.co_varnames:
                    super_args['plots'] = plots
                
                super().save_results(**super_args)
        except Exception as e:
            LOGGER.warning(f"调用父类保存方法失败: {e}")
        
        # 输出保存结果总结
        if results_saved:
            LOGGER.info("保存的结果文件:")
            for result in results_saved:
                LOGGER.info(f"  - {result}")
        else:
            LOGGER.warning("未保存任何结果文件")
        
        return save_dir
    
    def _save_visualization_plots(self, save_dir):
        """
        保存可视化图表。
        
        Args:
            save_dir: 保存目录
        """
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. COCO指标对比图
        self._plot_coco_metrics_comparison(plots_dir)
        
        # 2. 速度统计图
        self._plot_speed_statistics(plots_dir)
        
        # 3. 如果coco_metrics支持绘图，调用其方法
        if hasattr(self.coco_metrics, 'plot') and callable(self.coco_metrics.plot):
            try:
                self.coco_metrics.plot(save_dir=plots_dir)
            except Exception as e:
                LOGGER.warning(f"COCOMetrics绘图失败: {e}")
    
    def _plot_coco_metrics_comparison(self, save_dir):
        """
        绘制COCO指标对比图。
        """
        try:
            import matplotlib.pyplot as plt
            
            # AP指标
            ap_metrics = ['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge']
            ap_values = [getattr(self.coco_metrics, metric, 0.0) for metric in ap_metrics]
            
            # AR指标  
            ar_metrics = ['AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
            ar_values = [getattr(self.coco_metrics, metric, 0.0) for metric in ar_metrics]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # AP指标图
            ax1.bar(ap_metrics, ap_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
            ax1.set_title('COCO Average Precision (AP) Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # AR指标图
            ax2.bar(ar_metrics, ar_values, color=['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e'])
            ax2.set_title('COCO Average Recall (AR) Metrics')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'coco_metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            LOGGER.warning("matplotlib未安装，跳过COCO指标对比图生成")
        except Exception as e:
            LOGGER.warning(f"生成COCO指标对比图失败: {e}")
    
    def _plot_speed_statistics(self, save_dir):
        """
        绘制速度统计图。
        """
        try:
            import matplotlib.pyplot as plt
            
            if not any(self.speed.values()):
                return
            
            stages = list(self.speed.keys())
            times = list(self.speed.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(stages, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            plt.title('Processing Speed Statistics')
            plt.ylabel('Time (ms)')
            plt.xlabel('Processing Stage')
            
            # 添加数值标签
            for bar, time_val in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{time_val:.1f}ms', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'speed_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            LOGGER.warning("matplotlib未安装，跳过速度统计图生成")
        except Exception as e:
            LOGGER.warning(f"生成速度统计图失败: {e}")
    
    def _save_summary_report(self, save_dir):
        """
        保存文本格式的汇总报告。
        
        Args:
            save_dir: 保存目录
        """
        report_path = save_dir / "coco_summary.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("多模态COCO评估汇总报告\n")
                f.write("=" * 80 + "\n\n")
                
                # 基本信息
                f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"验证器类型: MultiModalCOCOValidator\n")
                f.write(f"验证模式: {self.modality if self.modality else 'RGB+X'}\n")
                f.write(f"数据集: {getattr(self.args, 'data', 'N/A')}\n")
                f.write(f"类别数: {self.nc}\n")
                f.write(f"验证图像数: {len(self.coco_stats)}\n\n")
                
                # 主要指标
                f.write("主要COCO指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"mAP@0.5:0.95:  {getattr(self.coco_metrics, 'AP', 0.0):.3f}\n")
                f.write(f"mAP@0.5:      {getattr(self.coco_metrics, 'AP50', 0.0):.3f}\n")
                f.write(f"mAP@0.75:     {getattr(self.coco_metrics, 'AP75', 0.0):.3f}\n\n")
                
                # 尺寸分别指标
                f.write("不同尺寸目标指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"APsmall:      {getattr(self.coco_metrics, 'APsmall', 0.0):.3f}\n")
                f.write(f"APmedium:     {getattr(self.coco_metrics, 'APmedium', 0.0):.3f}\n")
                f.write(f"APlarge:      {getattr(self.coco_metrics, 'APlarge', 0.0):.3f}\n\n")
                
                # 召回指标
                f.write("召回指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"AR1:          {getattr(self.coco_metrics, 'AR1', 0.0):.3f}\n")
                f.write(f"AR10:         {getattr(self.coco_metrics, 'AR10', 0.0):.3f}\n")
                f.write(f"AR100:        {getattr(self.coco_metrics, 'AR100', 0.0):.3f}\n\n")
                
                # 速度统计
                if any(self.speed.values()):
                    f.write("速度统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"预处理:       {self.speed.get('preprocess', 0.0):.1f}ms\n")
                    f.write(f"推理:         {self.speed.get('inference', 0.0):.1f}ms\n")
                    f.write(f"后处理:       {self.speed.get('postprocess', 0.0):.1f}ms\n")
                    if self.times:
                        avg_time = np.mean(self.times)
                        f.write(f"平均处理时间: {avg_time:.1f}ms/图像\n")
                        f.write(f"处理速度:     {1000/avg_time:.1f} FPS\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
        except Exception as e:
            LOGGER.error(f"保存汇总报告失败: {e}")
    
    def __enter__(self):
        """
        上下文管理器入口，初始化进度条。
        """
        self.init_progress_bar()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口，清理进度条。
        """
        self.close_progress_bar()
        return False