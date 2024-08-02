```bash
Filename: /home/xujg/code/UAV-VisionLoc-Deploy/python/main.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
   182     75.0 MiB     75.0 MiB           1   @profile
   183                                         def main():
   184                                         
   185     75.0 MiB      0.1 MiB           1       parser = argparse.ArgumentParser(description='Process some integers.')
   186                                             # basic params
   187     75.0 MiB      0.0 MiB           1       parser.add_argument('--model_path', type=str, default= "/home/xujg/code/UAV-VisionLoc-Deploy/model/uvl_731.rknn", help='model path, could be .pt or .rknn file')
   188                                         
   189     75.0 MiB      0.0 MiB           1       parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
   190     75.0 MiB      0.0 MiB           1       parser.add_argument('--img_save', action='store_true', default=True, help='save the result')
   191     75.0 MiB      0.0 MiB           1       parser.add_argument('--save_path', default="/home/xujg/code/UAV-VisionLoc-Deploy/python/result", help='save the result')
   192                                             # data params
   193     75.0 MiB      0.0 MiB           1       parser.add_argument('--img_folder', type=str, default='/home/xujg/code/UAV-VisionLoc-Deploy/data/queries', help='img folder path')
   194     75.0 MiB      0.0 MiB           1       parser.add_argument('--path_local_database', type=str, default='/home/xujg/code/UAV-VisionLoc-Deploy/data/database/database_features.h5', help='load local features and utms of database')
   195                                         
   196     75.0 MiB      0.0 MiB           2       parser.add_argument(
   197     75.0 MiB      0.0 MiB           1           "--features_dim",
   198     75.0 MiB      0.0 MiB           1           type=int,
   199     75.0 MiB      0.0 MiB           1           default=4096,
   200     75.0 MiB      0.0 MiB           1           help="NetVLAD output dims.",
   201                                             )
   202                                             # retrieval params
   203     75.0 MiB      0.0 MiB           2       parser.add_argument(
   204     75.0 MiB      0.0 MiB           1           "--recall_values",
   205     75.0 MiB      0.0 MiB           1           type=int,
   206     75.0 MiB      0.0 MiB           1           default=[1, 5, 10, 20],
   207     75.0 MiB      0.0 MiB           1           nargs="+",
   208     75.0 MiB      0.0 MiB           1           help="Recalls to be computed, such as R@5.",
   209                                             )
   210     75.0 MiB      0.0 MiB           2       parser.add_argument(
   211     75.0 MiB      0.0 MiB           1           "--use_best_n",
   212     75.0 MiB      0.0 MiB           1           type=int,
   213     75.0 MiB      0.0 MiB           1           default=1,
   214     75.0 MiB      0.0 MiB           1           help="Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1"
   215                                             )
   216                                         
   217     75.0 MiB      0.0 MiB           1       args = parser.parse_args()
   218                                         
   219    883.5 MiB    808.4 MiB           1       rk_engine = RKInfer(args)
   220                                         
   221                                             # init model
   222    929.7 MiB     46.2 MiB           1       rk_engine.setup_model()
   223                                         
   224    929.7 MiB      0.0 MiB           1       file_list = sorted(os.listdir(args.img_folder))
   225    929.7 MiB      0.0 MiB           1       img_list = []
   226    929.7 MiB      0.0 MiB           4       for path in file_list:
   227    929.7 MiB      0.0 MiB           3           if img_check(path):
   228    929.7 MiB      0.0 MiB           1               img_list.append(path)
   229                                         
   230                                             # run test
   231    944.5 MiB      0.3 MiB           2       for i in tqdm(range(len(img_list))):
   232    930.0 MiB      0.0 MiB           1           print('infer {}/{}'.format(i+1, len(img_list)), end='\r')
   233                                         
   234    930.0 MiB      0.0 MiB           1           img_name = img_list[i]
   235    930.0 MiB      0.0 MiB           1           img_path = os.path.join(args.img_folder, img_name)
   236    930.0 MiB      0.0 MiB           1           if not os.path.exists(img_path):
   237                                                     print("{} is not found", img_name)
   238                                                     continue
   239    930.0 MiB      0.0 MiB           1           t1 = time()
   240    936.4 MiB      6.4 MiB           1           input_data = pre_process(img_path)
   241    936.4 MiB      0.0 MiB           1           t2 = time()
   242    936.4 MiB      0.0 MiB           1           input_data = np.expand_dims(input_data, 0)
   243    944.2 MiB      7.8 MiB           1           position = rk_engine.model_inference(input_data)
   244    944.2 MiB      0.0 MiB           1           print("pre_time: {:.4f} s".format(t2-t1) )
   245    944.5 MiB      0.3 MiB           1           print("position: ", position)
   246                                                 # t3 = time()
   247                                         
   248                                         
   249                                         
   250    944.5 MiB      0.0 MiB           1           if args.img_show or args.img_save:
   251    944.5 MiB      0.0 MiB           1               print('\n\nIMG: {}'.format(img_name))
   252    944.5 MiB      0.0 MiB           1               img_p = cv2.imread(img_path)
   253    944.5 MiB      0.0 MiB           1               draw(img_p, position)
   254                                         
   255    944.5 MiB      0.0 MiB           1               if args.img_save:
   256    944.5 MiB      0.0 MiB           1                   if not os.path.exists(args.save_path):
   257                                                             os.mkdir(args.save_path)
   258    944.5 MiB      0.0 MiB           1                   result_path = os.path.join(args.save_path, img_name)
   259    944.5 MiB      0.0 MiB           1                   cv2.imwrite(result_path, img_p)
   260    944.5 MiB      0.0 MiB           1                   print('Position result save to {}'.format(result_path))
   261                                         
   262    944.5 MiB      0.0 MiB           1               if args.img_show:
   263                                                         cv2.imshow("full post process result", img_p)
   264                                                         cv2.waitKeyEx(0)
```

从 `memory_profiler` 的输出中，我们可以看到程序在不同代码行的内存使用情况。以下是详细的分析：

### 内存使用情况分析

主要内存在类实例化过程中加载local_database时所占用的内存，大约500M

1. **程序启动**：
    
    - 初始内存使用：75.0 MiB
    - 代码行：182
2. **解析命令行参数**：
    
    - 内存增量：0.1 MiB
    - 代码行：185
    - 说明：创建 `argparse.ArgumentParser` 对象，占用 0.1 MiB 内存。
3. **加载和设置模型**：
    
    - 内存增量：808.4 MiB
    - 代码行：219
    - 说明：创建 `RKInfer` 对象（`rk_engine`），加载模型文件并进行初始化，占用了大量内存。
4. **初始化模型**：
    
    - 内存增量：46.2 MiB
    - 代码行：222
    - 说明：`rk_engine.setup_model()` 函数调用，占用了 46.2 MiB 内存。
5. **处理图像文件列表**：
    
    - 内存增量：0.0 MiB
    - 代码行：224-228
    - 说明：读取图像文件夹内容，生成图像文件列表（`img_list`），此步骤内存使用无明显变化。
6. **推理循环**：
    
    - 内存增量：若干小增量，总计约 14.8 MiB
    - 代码行：231-263
    - 说明：对于每张图像，预处理、推理和后处理步骤略微增加了内存使用。
        - **预处理**：
            - 内存增量：6.4 MiB
            - 代码行：240
        - **推理**：
            - 内存增量：7.8 MiB
            - 代码行：243
        - **结果保存和显示**：
            - 内存增量：0.3 MiB
            - 代码行：245

### 性能瓶颈

1. **模型加载**：
    
    - 内存增量：808.4 MiB（代码行：219）
    - `RKInfer` 对象的创建和模型加载是内存使用的主要瓶颈。
2. **模型初始化**：
    
    - 内存增量：46.2 MiB（代码行：222）
    - `setup_model` 函数调用也占用了较多内存。
3. **图像预处理和推理**：
    
    - 内存增量：分别为 6.4 MiB 和 7.8 MiB（代码行：240 和 243）
    - 对每张图像的处理略微增加了内存使用，但在大数据量下累积起来也可能显著。

### 待优化

1. **优化模型加载**：
    
    - 如果可能，优化 `RKInfer` 的模型加载和初始化过程，减少不必要的内存占用。
2. **批量处理图像**：
    
    - 考虑批量预处理和推理图像，减少每次循环中内存的反复分配和释放。
3. **检查内存泄漏**：
    
    - 确保每次循环中创建的临时对象在不再使用时被正确释放，避免内存泄漏。
4. **优化数据结构**：
    
    - 如果图像列表和预处理步骤中使用的数据结构可以优化（例如使用更高效的数据存储方式），可以进一步减少内存使用。

通过这些分析和优化建议，可以有效地提升程序的内存效率和整体性能。

