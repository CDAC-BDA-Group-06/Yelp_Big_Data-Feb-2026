# ─────────────────────────────────────────────
# BRONZE BUCKET  (raw JSON from Kaggle)
# NOTE: Encryption config and Public Access Block are managed by AWS Academy
# default account policy. LabRole denies s3:GetEncryptionConfiguration and
# s3:GetBucketPublicAccessBlock so these resources cannot be managed by Terraform.
# ─────────────────────────────────────────────
resource "aws_s3_bucket" "bronze" {
  bucket        = var.bronze_bucket_name
  force_destroy = true
}

# ─────────────────────────────────────────────
# SILVER BUCKET  (cleaned Parquet from Glue)
# ─────────────────────────────────────────────
resource "aws_s3_bucket" "silver" {
  bucket        = var.silver_bucket_name
  force_destroy = true
}

# ─────────────────────────────────────────────
# GOLD BUCKET  (analytics BI/ML/RAG Parquet from Glue)
# ─────────────────────────────────────────────
resource "aws_s3_bucket" "gold" {
  bucket        = var.gold_bucket_name
  force_destroy = true
}



