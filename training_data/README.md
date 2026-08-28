# Local Filter Training Data

This folder stores human review labels used to train the local competitor evidence filter.

Default label file:

```text
training_data/review_labels.csv
```

Required column:

```text
human_label
```

Recommended search-card columns:

```text
product_category
product_type_key
product_type_label
search_card_candidate
```

Supported labels:

```text
include
exclude
verify_later
```

Recommended workflow:

1. Run a collection job.
2. Review `问题页面核验清单.csv`, `人工抽样标注表.csv`, and `所有采集来源.csv`.
3. Copy useful rows into `training_data/review_labels.csv`, or copy `training_data/review_labels.example.csv` as the first local label file.
4. Fill `human_label` and `human_reason`.
5. Keep the product type columns from `training_review_sample.csv` if they exist. They let the tool turn the reviewed rows into reusable search cards for the same kind of product.
6. Train the model from the UI or run:

```bash
python3 train_filter_model.py --labels training_data/review_labels.csv --model-out models/filter_model.pt
```

The trained model is a local `.pt` checkpoint. It improves the harvester's future inclusion, exclusion, and verification scores without binding the project to one LLM provider.

Training also writes product search cards to `search_cards/` by default. A card stores the terms, reusable source domains, page roles, and exclusion signals learned from human review for one product type. By default, 3 labeled rows in one product type are enough to create a low-confidence card. Rows without product type metadata still train the filter model, but they do not create search cards. The next collection run loads matching cards automatically before searching, so the tool can handle more product categories over time without prebuilding every category by hand.
