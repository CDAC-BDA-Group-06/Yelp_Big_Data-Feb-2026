# Trigger run with updated AWS Academy session credentials
terraform {
  cloud {
    organization = "cdac-bda-group06"

    workspaces {
      name = "yelp-bigdata-workspace"
    }
  }
}

