terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Artifact Registry for AeroEval Docker Images
resource "google_artifact_registry_repository" "aeroeval_repo" {
  location      = var.region
  repository_id = "aeroeval-docker-repo"
  description   = "Docker repository for AeroEval agent container"
  format        = "DOCKER"
}

# Dedicated Service Account for AeroEval Agent in Cloud Run
resource "google_service_account" "aeroeval_sa" {
  account_id   = "aeroeval-agent-sa"
  display_name = "AeroEval Cloud Run Service Account"
}

# Grant Vertex AI User role to the Service Account for Gemini & Claude invocation
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.aeroeval_sa.email}"
}

# Grant Firestore User role to the Service Account for session memory persistence
resource "google_project_iam_member" "firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.aeroeval_sa.email}"
}

# Grant Storage Object Viewer role to the Service Account for telemetry data
resource "google_project_iam_member" "storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.aeroeval_sa.email}"
}

# Grant Sensitive Data Protection (DLP) User role to the Service Account for PII redaction
resource "google_project_iam_member" "dlp_user" {
  project = var.project_id
  role    = "roles/dlp.user"
  member  = "serviceAccount:${google_service_account.aeroeval_sa.email}"
}

# Google Cloud Storage Bucket for Telemetry Data
resource "google_storage_bucket" "telemetry_bucket" {
  name                        = var.gcs_bucket_name
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = false
}

# Automatically seed the GCS bucket with telemetry datasets and registry from data/
resource "google_storage_bucket_object" "telemetry_data_files" {
  for_each = fileset("${path.module}/../data", "**/*")
  name     = each.value
  bucket   = google_storage_bucket.telemetry_bucket.name
  source   = "${path.module}/../data/${each.value}"
}

# Firestore Native Database for AeroEval Context & Session Memory
resource "google_firestore_database" "session_db" {
  project                 = var.project_id
  name                    = var.firestore_database_name
  location_id             = var.firestore_location_id
  type                    = "FIRESTORE_NATIVE"
  delete_protection_state = "DELETE_PROTECTION_DISABLED"
  deletion_policy         = "DELETE"
}

# Cloud Run Service hosting the AeroEval Agent
resource "google_cloud_run_v2_service" "aeroeval_service" {
  name     = "aeroeval-agent"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.aeroeval_sa.email

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "1000m"
          memory = "1Gi"
        }
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "VERTEX_LOCATION"
        value = "global"
      }
      env {
        name  = "VERTEX_CLAUDE_LOCATION"
        value = "us-east5"
      }
      env {
        name  = "USE_FIRESTORE"
        value = "true"
      }
      env {
        name  = "FIRESTORE_DATABASE"
        value = var.firestore_database_name
      }
      env {
        name  = "FIRESTORE_COLLECTION"
        value = "aeroeval_sessions"
      }
      env {
        name  = "GCS_BUCKET_NAME"
        value = var.gcs_bucket_name
      }
    }
  }

  depends_on = [
    google_firestore_database.session_db,
    google_storage_bucket.telemetry_bucket,
    google_storage_bucket_object.telemetry_data_files,
  ]
}

# Allow public unauthenticated invocations for assessment evaluation
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.aeroeval_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
