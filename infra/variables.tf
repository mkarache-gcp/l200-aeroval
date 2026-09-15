variable "project_id" {
  description = "The Google Cloud Project ID to deploy resources to."
  type        = string
  default     = "onboardingproject-507522"
}

variable "region" {
  description = "The Google Cloud Region for Cloud Run."
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "The full URI of the container image to deploy."
  type        = string
  default     = "us-central1-docker.pkg.dev/onboardingproject-507522/aeroeval-docker-repo/aeroeval:latest"
}

variable "firestore_database_name" {
  description = "The database name for Firestore session store."
  type        = string
  default     = "aeroeval"
}

variable "firestore_location_id" {
  description = "The multi-region or regional location for Firestore (e.g., 'nam5' for US multi-region or 'us-central1')."
  type        = string
  default     = "nam5"
}

variable "gcs_bucket_name" {
  description = "The Google Cloud Storage bucket name for flight telemetry data."
  type        = string
  default     = "onboardingproject-507522-aeroeval-data"
}

