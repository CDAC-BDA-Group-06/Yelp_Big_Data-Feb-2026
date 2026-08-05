# Terraform Cloud Backend configuration for CDAC BDA Group 06
terraform {
  cloud {
    organization = "cdac-bda-group06"

    workspaces {
      name = "yelp-production-workspace"
    }
  }
}
