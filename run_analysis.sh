
#!/usr/bin/env bash
#
# ================================================================
# Script to execute the complete analysis of gravitational lensing 
# ================================================================
#
# Execution order:
# 1. model training and saving
# 2. model testing
# 3. errors calculation and plots
#

GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
RED=$'\033[0;31m'
NC=$'\033[0m'

# Show help message for the script.
show_help() {
    cat << EOF
    ${CYAN}=========================================${NC}
    ${CYAN}   Gravitational Lensing Analysis Script   ${NC}
    ${CYAN}=========================================${NC}

    ${GREEN}Description:${NC}
        This script automates the execution of the analysis of gravitational lensing.

    ${YELLOW}Usage:${NC} ./run_analysis.sh [OPTIONS]

    ${YELLOW}Options:${NC}
      -strain, --skip-train         Skip model training step.
      -stest, --skip-test          Skip model testing step.
      -sres, --skip-results       Skip results calculation step.
      --config=VALUE       Set the 'prueba' parameter in main_config.ini to VALUE.
      -h, --help           Show this help message and exit.
    
    ${YELLOW}Example:${NC}
        ./run_analysis.sh --skip-train --config=test1
    
    ${YELLOW}Note:${NC}
        Ensure that you have the necessary permissions to create directories and write log files in the specified paths.
EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

PYTHON_BIN=python
MAIN_PATH="./csst_catalog/test18/"
LOG_DIR="$MAIN_PATH/logs"
DATE=$(date +"%Y_%m_%d_%H_%M_%S")
LOG_FILE="$LOG_DIR/analysis_$DATE.log"

# Function to check and create directory if it doesn't exist
check_or_create_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "creating directory: $dir" | tee -a "$LOG_FILE"
        mkdir -p "$dir"
        if [ $? -ne 0 ]; then
            echo "Error creating directory: $dir" | tee -a "$LOG_FILE"
            exit 1
        fi
    fi
}

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
check_or_create_dir "$MAIN_PATH"
check_or_create_dir "$LOG_DIR"

echo "${YELLOW}===============================${NC}" | tee -a $LOG_FILE
echo "${YELLOW}Starting analysis at $(date)${NC}" | tee -a $LOG_FILE
echo "${YELLOW}===============================${NC}" | tee -a $LOG_FILE

# Set README file
README_FILE="${MAIN_PATH}alexnet_1/README_analysis.txt"
echo "${YELLOW}Analysis started at $(date)${NC}" > "$README_FILE"
echo "${YELLOW}Log file: $LOG_FILE${NC}" >> "$README_FILE"

# Skip model training flag: false by default --skip-train
SKIP_TRAIN=false
for arg in "$@"; do
  case $arg in
    -strain | --skip-train)
      SKIP_TRAIN=true
      shift
      ;;
    *)
      ;;
  esac
done

# Skip model testing flag: false by default --skip-test
SKIP_TEST=false
for arg in "$@"; do
  case $arg in
    -stest | --skip-test)
      SKIP_TEST=true
      shift
      ;;
    *)
      ;;
  esac
done

# Skip results calculation flag: false by default --skip-results
SKIP_RESULTS=false
for arg in "$@"; do
  case $arg in
    -sres | --skip-results)
      SKIP_RESULTS=true
      shift
      ;;
    *)
      ;;
  esac
done

CONFIG_VALUE=""
for arg in "$@"; do
    case $arg in
        --config=*)
        CONFIG_VALUE="${arg#*=}"
        shift
        ;;
    esac
done

if [ -n "$CONFIG_VALUE" ]; then
    sed -i '' "s/prueba *= *.*/prueba = ${CONFIG_VALUE}/" main_config.ini
fi

echo "Applying test: $CONFIG_VALUE" | tee -a $LOG_FILE

# Execute training and saving model
if [ "$SKIP_TRAIN" = false ]; then
    echo "\n[1/4] Running "model"_test.py for model training..." | tee -a $LOG_FILE
    $PYTHON_BIN alexnet_test.py >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "${RED}Error during model training. Check the log file for details.${NC}" | tee -a $LOG_FILE
        exit 1
    else
        echo "${GREEN}Model training completed successfully.${NC}" | tee -a $LOG_FILE
    fi
else
    echo "\n[1/4] Skipping model training." | tee -a $LOG_FILE
fi

# Execute test.py
if [ "$SKIP_TEST" = false ]; then
    echo "\n[2/4] Running test.py for model testing..." | tee -a $LOG_FILE
    $PYTHON_BIN test.py >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "${RED}Error during model testing. Check the log file for details.${NC}" | tee -a $LOG_FILE
        exit 1
    else
        echo "${GREEN}Model testing completed successfully.${NC}" | tee -a $LOG_FILE
    fi
else
    echo "\n[2/4] Skipping model testing." | tee -a $LOG_FILE
fi

# Execute results_paper.py
if [ "$SKIP_RESULTS" = false ]; then
    echo "\n[3/4] Running results_paper.py for results..." | tee -a $LOG_FILE
    $PYTHON_BIN results_paper.py >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "${RED}Error during results calculation. Check the log file for details.${NC}" | tee -a $LOG_FILE
        exit 1  
    else
        echo "${GREEN}Results calculation completed successfully.${NC}" | tee -a $LOG_FILE
    fi
else
    echo "\n[3/4] Skipping results calculation." | tee -a $LOG_FILE
fi

# Exexute PSNR comparison compare_images.py
echo "\n[4/4] Running compare_images.py for PSNR comparison..." | tee -a $LOG_FILE
$PYTHON_BIN compare_images.py >> $LOG_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "${RED}Error during PSNR comparison. Check the log file for details.${NC}" | tee -a $LOG_FILE
    exit 1  
else    
    echo "${GREEN}PSNR comparison completed successfully.${NC}" | tee -a $LOG_FILE
fi

echo "\n${YELLOW}===============================${NC}" | tee -a $LOG_FILE
echo "${YELLOW}Analysis completed at $(date)${NC}" | tee -a $LOG_FILE
echo "${YELLOW}===============================${NC}" | tee -a $LOG_FILE  
echo "${GREEN}Log file saved to $LOG_FILE${NC}"
