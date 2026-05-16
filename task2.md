Đánh vào hạ tầng cơ sở dân sinh là triệt hạ đường sống của người dân vô tội đó, thưa ngài . Thích Thích Vui Ngạc nhiên Buồn 37 37 Trả lời Báo vi phạm 5h trước
score=0.562 · pseudo=1 · https://vnexpress.net/toi-hau-thu-sap-het-han-ong-trump-de-doa-xoa-so-nen-van-minh-iran-5059695.html

Dân thường họ có tội tình gì mà đòi xóa sổ cuộc sống của họ chứ Thích Thích Vui Ngạc nhiên Buồn 18 17 1 Trả lời Báo vi phạm 4h trước
score=0.521 · pseudo=1 · https://vnexpress.net/toi-hau-thu-sap-het-han-ong-trump-de-doa-xoa-so-nen-van-minh-iran-5059695.html

Tối ưu lại prompt để đỡ ngốn token, các file liên quan sẽ là như /Users/mac/git/Thesis/comment_crawl.py và /Users/mac/git/Thesis/setup_and_crawl.py
Hiện tại có hai điểm ta cần cải thiện ở đây:
1. Crawl ở comment section đang kéo theo một số thông tin không cần thiết như "37 37 Trả lời Báo vi phạm 5h trước" hay "18 17 1 Trả lời Báo vi phạm 4h trước" bạn có thể thấy ví dụ trên.
2. Hiện tại pseudo=1 là đang gán theo score là toxic tầm 0.562%, nhưng nếu không phải toxic thì tôi có thể dùng cơ chế nào để cho model học được dataset mới một cách chuẩn hơn không? Ví dụ khi accept thì có accept theo pseudo 1 hay 0, kiểu vậy. 