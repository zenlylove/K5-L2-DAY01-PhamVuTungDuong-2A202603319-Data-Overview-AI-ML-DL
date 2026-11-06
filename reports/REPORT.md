# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics: Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** 'yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt'

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh chứ không phải một vật thể riêng biệt. Vì mô hình Image Classification phân tích tổng thể nội dung hình ảnh và đưa ra lớp có khả năng phù hợp nhất. Trong ảnh traffic có nhiều phương tiện như xe buýt, ô tô và các phương tiện khác, nhưng mô hình chọn cab làm lớp có điểm số cao nhất.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp được cung cấp sẵn và đi kèm với taxonomy/checkpoint đã được huấn luyện. Trong bài thực hành này, mô hình sử dụng ImageNet-1K gồm 1.000 lớp được định nghĩa trước. Model không tự nghĩ ra tên lớp mới mà lựa chọn từ danh sách có sẵn.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả ba thông tin để xác định chính xác nhãn:

class_id: Mã số định danh của lớp.

class_name: Tên của lớp giúp con người dễ đọc và hiểu.

taxonomy_name: Tên bộ danh mục quy định các lớp được sử dụng.

Việc lưu đầy đủ giúp tránh nhầm lẫn giữa các bộ taxonomy khác nhau, thuận tiện cho việc quản lý, đối chiếu và sử dụng dữ liệu trong quá trình huấn luyện mô hình.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ mục tiêu phân loại, tiêu chí xác định chủ đề chính và cách lựa chọn lớp khi ảnh chứa nhiều đối tượng khác nhau.

Ví dụ: Nếu yêu cầu phân loại chủ đề chính thì annotator phải lựa chọn lớp phản ánh nội dung chính của bức ảnh, không tùy ý chọn một vật thể xuất hiện trong ảnh. Nếu không thể xác định lớp phù hợp thì cần chuyển reviewer kiểm tra.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ thể hiện mức độ tin cậy của mô hình đối với kết quả dự đoán, không chứng minh rằng dự đoán đó chính xác.

Ground truth phải được xác định dựa trên hình ảnh thực tế, guideline và quá trình gán nhãn của con người, sau đó được reviewer kiểm tra.

Ví dụ: Lớp cab có score 0.510915 không có nghĩa đây là nhãn đúng hoặc ground truth của bức ảnh.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
Chọn đối tượng person trong ảnh kitchen:
{
  "sample_id": "kitchen",
  "class_name": "person",
  "score": 0.912625,
  "bbox_xyxy": [385.33, 69.24, 498.92, 348.92],
  "bbox_width": 113.58,
  "bbox_height": 279.68
}
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên bên trái: tọa độ (385.33, 69.24).
- Góc dưới bên phải: tọa độ (498.92, 348.92).
- Chiều rộng: 113.59 pixel.
- Chiều cao: 279.68 pixel.
Bounding box là một vùng hình chữ nhật xác định vị trí và kích thước của đối tượng person trong ảnh kitchen. Góc trên bên trái và góc dưới bên phải xác định phạm vi của hình chữ nhật bao quanh đối tượng.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Dữ liệu bạn gửi hiện tại chỉ chứa kết quả với ngưỡng 0.35.
Ngưỡng	Số prediction
0.20	Chưa có dữ liệu
0.35	11
0.60	Chưa có dữ liệu
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Reviewer phải kiểm tra nhiều prediction hơn, loại bỏ các hộp nhận diện sai, kiểm tra vật thể bị nhận diện nhầm và phát hiện các box trùng lặp hoặc không chính xác.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Người gán nhãn phải vẽ bounding box bao quanh toàn bộ phần vật thể nhìn thấy trong ảnh, không bao gồm quá nhiều vùng nền và không bỏ sót phần vật thể quan sát được.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong dữ liệu JSON, có một số box nằm sát mép ảnh, ví dụ box oven có tọa độ x bắt đầu là 0.11 pixel. Tuy nhiên, JSON chưa đủ thông tin để kết luận vật thể bị che khuất hay bị cắt mép. Cần xem ảnh detection_predictions.png để xác nhận.

Nếu trường hợp đó không được guideline quy định rõ, cần escalation lên Lab Coach thay vì tự ý đưa ra quy tắc.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
Chọn đối tượng bowl có instance_id = kitchen-010.
{
  "sample_id": "kitchen",
  "instance_id": "kitchen-010",
  "class_id": 45,
  "class_name": "bowl",
  "score": 0.372406,
  "coordinate_unit": "pixel",
  "bbox_format": "xyxy",
  "bbox_xyxy": [
    154.46,
    169.34,
    182.99,
    184.98
  ],
  "polygon_point_count": 12,
  "polygon_xy": [
    [155, 170],
    [155, 177],
    [156, 178],
    [156, 179],
    [161, 184],
    [176, 184],
    [180, 180],
    [180, 179],
    [181, 178],
    [181, 177],
    [182, 176],
    [182, 170]
  ]
}
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Bounding Box: Là hình chữ nhật bao quanh đối tượng, được xác định bởi bốn tọa độ [x1, y1, x2, y2].
- Polygon: Là tập hợp các điểm tọa độ mô tả đường viền của đối tượng, giúp xác định vùng hình ảnh thuộc về đối tượng đó.
- Mask: Là mặt nạ phân đoạn, thể hiện các pixel thuộc về một đối tượng cụ thể.
Trong ví dụ kitchen-010, Bounding Box xác định vị trí chiếc bát, còn Polygon gồm 12 điểm tọa độ mô tả đường viền của chiếc bát.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Đối với đối tượng kitchen-010:
- Số điểm Polygon: 12 điểm.
- Đơn vị tọa độ: pixel.
- Định dạng Bounding Box: xyxy.
- Kích thước ảnh: 640 × 427 pixel.
Các tọa độ Polygon được biểu diễn dưới dạng các cặp [x, y], trong đó x là tọa độ theo chiều ngang và y là tọa độ theo chiều dọc của ảnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Tiêu chí	            Object Detection	        Instance Segmentation
Mục tiêu	            Phát hiện đối tượng	        Phát hiện và phân vùng từng đối tượng
Kết quả	                Bounding Box	            Bounding Box và Mask
Đường viền	            Hình chữ nhật	            Theo hình dạng đối tượng
Mức độ chi tiết	        Xác định vị trí	            Xác định vị trí và vùng pixel
Ứng dụng	            Đếm và nhận diện vật thể	Phân tích hình dạng, diện tích và vùng đối tượng

Instance Segmentation cung cấp thông tin chi tiết hơn Object Detection vì không chỉ xác định vị trí mà còn xác định vùng pixel của từng đối tượng riêng biệt.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Vị trí: Chiếc bát màu hồng ở góc dưới bên trái ảnh, nơi bát tiếp xúc với mặt bàn và các vật thể xung quanh.

Lựa chọn: Guideline (áp dụng hướng dẫn).

Lý do: Đây là trường hợp các vật thể nằm sát nhau hoặc bị che khuất một phần. Áp dụng guideline bằng cách chỉ đánh dấu phần đối tượng có thể nhìn thấy, giữ đường biên theo hình dạng thực tế và không tự suy đoán phần bị che khuất. Chỉ escalation khi guideline không đủ để giải quyết trường hợp mơ hồ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

Tác vụ	Ground truth trông như thế nào? (đơn vị nhãn) \\	Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON \\	Annotator làm gì?	\\ Reviewer xem gì?
Phân loại ảnh (Image Classification)      \\	Mỗi ảnh có một hoặc nhiều nhãn lớp (class label), ví dụ: kitchen, traffic hoặc các lớp đối tượng theo yêu cầu bài toán.  \\ 	Ảnh có thể chứa nhiều đối tượng hoặc nhiều ngữ cảnh, dễ nhầm lớp chính. Prediction chỉ thể hiện kết quả mô hình, chưa đủ để kết luận nhãn đúng hay sai nếu không có Ground truth. \\	Gán nhãn đúng theo nội dung và guideline; chọn đúng lớp, không suy đoán khi ảnh không đủ thông tin. \\	Kiểm tra nhãn có phù hợp nội dung ảnh không; phát hiện ảnh gán sai lớp, thiếu nhãn hoặc nhãn không nhất quán.

Phát hiện vật thể (Object Detection) \\ 	Mỗi đối tượng có một nhãn lớp (class_name) và một Bounding Box gồm 4 tọa độ [x1, y1, x2, y2] theo pixel. \\ 	JSON ảnh kitchen có đối tượng person với score = 0.912625, bbox_xyxy = [385.33, 69.24, 498.92, 348.92]. Chiều rộng tính từ tọa độ là 113.59 px, trong khi trường bbox_width ghi 113.58 px, có sai khác 0.01 px do làm tròn hoặc cách lưu số. \\ 	Vẽ Bounding Box bao trọn đối tượng nhìn thấy; gán đúng class, không bỏ sót vật thể và không bao gồm quá nhiều nền. \\ 	Kiểm tra class có đúng không, Bounding Box có bao sát đối tượng không, có bỏ sót, nhận diện nhầm hoặc tạo box trùng lặp không; kiểm tra tính nhất quán của tọa độ.


Instance Segmentation \\ 	Mỗi đối tượng có nhãn lớp (class_name) và một Mask biểu diễn vùng pixel thuộc đối tượng; có thể lưu dưới dạng Polygon gồm các điểm [x, y]. \\ 	JSON kitchen-010 có class_name = bowl, score = 0.372406, polygon_point_count = 12. Confidence khá gần ngưỡng 0.35. Đối tượng nhỏ hoặc nằm sát vật khác khiến đường biên Mask khó xác định. Ảnh PNG cũng cho thấy nhiều đối tượng ở gần nhau. \\ 	Vẽ Mask theo đường biên phần đối tượng nhìn thấy; tách riêng từng instance cùng loại; áp dụng guideline với vùng bị che khuất, không tự suy đoán phần không nhìn thấy. \\ 	Kiểm tra Mask có bám sát đường viền không, có lấn sang nền hoặc đối tượng khác không, có bỏ sót vùng nhìn thấy và phân biệt đúng từng instance không.


## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu và hình ảnh được cung cấp trong phạm vi bài lab, không tự ý chia sẻ, sao chép hoặc sử dụng dữ liệu cho mục đích khác.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý, không tự ý sử dụng dữ liệu và báo cho giảng viên hoặc người phụ trách bài lab để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
