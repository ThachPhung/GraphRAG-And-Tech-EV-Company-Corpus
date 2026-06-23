# Lab Day 19: GraphRAG với Tech/EV Company Corpus

Project này hoàn thiện các phần trong lab: research, environment setup, indexing, graph construction, querying, và evaluation Flat RAG vs GraphRAG. Dataset đã được đặt trong `dataset/` với 70 tài liệu `.txt`.

## 1. Mục Tiêu

- Trích xuất thực thể và quan hệ từ corpus ngành xe điện Mỹ.
- Xây dựng knowledge graph bằng `NetworkX`.
- Truy vấn theo kiểu GraphRAG bằng cách tìm entity trong câu hỏi và duyệt graph 2-hop.
- So sánh với Flat RAG dùng TF-IDF retrieval.
- Xuất file cho Neo4j/Gephi để trực quan hóa.

## 2. Research Trả Lời Nhanh

**Entity Extraction**: LLM hoặc rule-based extractor nhận diện các thực thể như công ty, địa điểm, chính sách, công nghệ, thị trường. Trong project này, nếu có `OPENAI_API_KEY`, extractor dùng OpenAI để sinh triples; nếu không có key, code dùng rule-based extractor để vẫn chạy offline.

**Graph Construction**: Khử trùng lặp quan trọng vì cùng một công ty như `Tesla` hoặc `United States` xuất hiện ở nhiều tài liệu. Nếu không normalize và deduplicate, graph bị phân mảnh thành nhiều node tương đương, làm truy vấn nhiều bước kém chính xác.

**Query Answering**: Flat RAG tìm tài liệu gần câu hỏi nhất theo vector/text similarity. GraphRAG tìm entity chính, duyệt các node liên quan trong 2-hop, rồi textualize các cạnh/evidence để trả lời. BFS phù hợp khi cần đi theo quan hệ rõ ràng giữa entity; vector search phù hợp khi câu hỏi gần với một đoạn văn cụ thể.

## 3. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tuỳ chọn dùng LLM:

```bash
cp .env.example .env
# điền OPENAI_API_KEY trong .env
```

Nếu yêu cầu bài nộp bắt buộc dùng API, hãy chạy bằng OpenAI mode:

```bash
python src/build_index.py --use-openai
python src/evaluate.py
python src/visualize_graph.py
```

Khi chạy `--use-openai`, project sẽ gọi API để trích xuất triples và ghi token/cost vào `outputs/build_metrics.json`. Nếu chưa có `OPENAI_API_KEY`, script sẽ dừng lỗi để tránh vô tình nộp bản offline.

## 4. Build Index

Chạy offline:

```bash
python src/build_index.py
```

Chạy có OpenAI để trích xuất triples tốt hơn:

```bash
python src/build_index.py --use-openai
```

Chi phí ước tính mặc định dùng giá cấu hình trong `.env.example`:

```text
OPENAI_INPUT_COST_PER_1M=0.15
OPENAI_OUTPUT_COST_PER_1M=0.60
```

Nếu bảng giá model thay đổi, cập nhật 2 biến này trong `.env` rồi build lại.

Output chính trong `outputs/`:

- `triples.csv`: các bộ ba `(subject, predicate, object)` kèm evidence.
- `knowledge_graph.json`: graph để query bằng Python.
- `knowledge_graph.gexf`: mở bằng Gephi hoặc tool visualize graph.
- `flat_retriever.pkl`: index Flat RAG.
- `neo4j_nodes.csv` và `neo4j_relationships.csv`: CSV import vào Neo4j.

## 5. Querying

Ví dụ câu hỏi:

```bash
python src/query.py "Why did US EV sales growth slow in Q1 2024 and how was Tesla involved?"
python src/query.py "How do charging infrastructure concerns affect EV adoption in the United States?"
python src/query.py "How are Nikola's hydrogen truck strategy and partners connected?"
```

Nếu muốn LLM viết câu trả lời cuối:

```bash
python src/query.py "Which companies reported strong EV sales growth in Q1 2024?" --use-openai
```

## 6. Evaluation

Chạy 5 câu hỏi phức tạp như yêu cầu lab:

```bash
python src/evaluate.py
```

Kết quả được lưu tại:

```text
outputs/evaluation_results.csv
```

Trong file này có cột `note` để bạn đánh dấu trường hợp Flat RAG bị thiếu ngữ cảnh hoặc hallucinate, còn GraphRAG trả lời đúng nhờ liên kết entity.

## 7. Visualization

Tạo ảnh preview graph:

```bash
python src/visualize_graph.py
```

Ảnh sẽ nằm ở:

```text
outputs/graph_preview.png
```

Với Neo4j Desktop, có thể import `outputs/neo4j_nodes.csv` và `outputs/neo4j_relationships.csv`, hoặc dùng `knowledge_graph.gexf` với Gephi để xem graph trực quan.

## 8. Gợi Ý Báo Cáo So Sánh

- Flat RAG mạnh khi câu hỏi bám sát một bài cụ thể, ví dụ hỏi số liệu Q1 2024 từ Cox Automotive.
- GraphRAG mạnh hơn khi cần nối nhiều thực thể, ví dụ Tesla ảnh hưởng thị trường ra sao, chính sách liên quan đến adoption thế nào, hoặc Nikola liên kết với HYLA, Voltera, Iveco, Walmart/Linde trong chiến lược hydrogen ra sao.
- Hạn chế của bản offline là relation extraction dựa trên rule nên predicate có thể chưa giàu ngữ nghĩa bằng LLM. Bật `--use-openai` để triples chính xác hơn.
