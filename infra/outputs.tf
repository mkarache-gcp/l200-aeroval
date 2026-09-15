output "cloud_run_url" {
  description = "The public HTTPS URL of the deployed AeroEval Cloud Run service."
  value       = google_cloud_run_v2_service.aeroeval_service.uri
}

output "artifact_registry_repo" {
  description = "The name of the created Artifact Registry repository."
  value       = google_artifact_registry_repository.aeroeval_repo.name
}

output "firestore_database_name" {
  description = "The name of the provisioned Firestore database."
  value       = google_firestore_database.session_db.name
}

output "gcs_bucket_name" {
  description = "The name of the provisioned GCS bucket for telemetry data."
  value       = google_storage_bucket.telemetry_bucket.name
}

