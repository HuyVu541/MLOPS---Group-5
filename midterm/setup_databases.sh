#!/bin/bash

# Define variables
PG_USER="huyvu"
PG_PASSWORD="password"
DB_A="raw_data"
DB_B="feature_db"
PG_HOST="localhost"
PG_PORT="5432"

# Function to execute SQL commands as postgres user
execute_sql() {
  sudo -u postgres psql -c "$1"
}

# Update packages and install PostgreSQL if it's not installed
echo "Checking for PostgreSQL installation..."
if ! command -v psql &> /dev/null
then
  echo "PostgreSQL not found. Installing PostgreSQL..."
  sudo apt update
  sudo apt install -y postgresql postgresql-contrib
else
  echo "PostgreSQL is already installed."
fi

# Start PostgreSQL service if not already running
echo "Starting PostgreSQL service..."
sudo systemctl start postgresql

# Enable PostgreSQL to start on boot
sudo systemctl enable postgresql

# Set up the user and password for PostgreSQL
echo "Setting up PostgreSQL user and password..."

# Check if user already exists, and create if necessary
execute_sql "SELECT 1 FROM pg_roles WHERE rolname='$PG_USER'" || execute_sql "CREATE USER $PG_USER WITH PASSWORD '$PG_PASSWORD';"

# Grant superuser privileges to the user (optional, adjust as needed)
execute_sql "ALTER USER $PG_USER WITH SUPERUSER;"

# Create databases A and B
echo "Creating databases $DB_A and $DB_B..."
execute_sql "CREATE DATABASE $DB_A;"
execute_sql "CREATE DATABASE $DB_B;"

# Grant privileges to the user on databases
echo "Granting privileges to user $PG_USER on databases $DB_A and $DB_B..."
execute_sql "GRANT ALL PRIVILEGES ON DATABASE $DB_A TO $PG_USER;"
execute_sql "GRANT ALL PRIVILEGES ON DATABASE $DB_B TO $PG_USER;"

# Confirming the setup
echo "Setup complete! PostgreSQL user '$PG_USER' with password '$PG_PASSWORD' has been created."
echo "Databases '$DB_A' and '$DB_B' have been created and privileges granted to '$PG_USER'."

# Optionally, you can log in to the database as the created user
echo "You can now login as the user $PG_USER with: psql -U $PG_USER -d $DB_A"
