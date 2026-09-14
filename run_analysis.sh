#!/usr/bin/env bash
#
# ==============================================================================
# Gravitational Lensing Analysis Orchestrator
# ==============================================================================
# Master entry point script for CNN model testing, paper error calculations,
# and image reconstruction comparisons (PSNR & SSIM).
#
# All parameters are centrally managed in `main_config.ini`. Command-line arguments
# allow overriding key settings dynamically.
# ==============================================================================

GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
RED=$'\033[0;31m'
NC=$'\033[0m'

show_help() {
    cat << EOF
    ${CYAN}================================================================${NC}
    ${CYAN}             Gravitational Lensing Analysis Orchestrator        ${NC}
    ${CYAN}================================================================${NC}

    ${GREEN}Description:${NC}
        Automates testing of pre-trained CNN models (direct vs deepensemble),
        error analysis (results_paper.py), and image comparison metrics (PSNR & SSIM).

    ${YELLOW}Usage:${NC} ./run_analysis.sh [OPTIONS]

    ${YELLOW}Options:${NC}
      --mode=MODE              Select test mode: 'direct' or 'deepensemble'.
      --config=VALUE           Set the 'prueba' parameter in main_config.ini.
      --prueba=VALUE           Alias for --config=VALUE.
      --model-dir=PATH         Specify path to model directory or Kaggle run output.
      -strain, --skip-train    Skip model training step.
      -stest, --skip-test     Skip model testing step.
      -sres, --skip-results    Skip results calculation step (results_paper.py).
      -scomp, --skip-compare   Skip image comparison & SSIM/PSNR evaluation step.
      -h, --help               Show this help message and exit.

    ${YELLOW}Examples:${NC}
        # Run direct model test with configuration test1
        ./run_analysis.sh --skip-train --mode=direct --config=test1

        # Run Deep Ensemble test using pre-trained models from Kaggle
        ./run_analysis.sh --skip-train --mode=deepensemble --config=alexnet_new --model-dir=./kaggle_runs/output/alexnet/alexnet_new

EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

PYTHON_BIN=python

# Default Flags
SKIP_TRAIN=false
SKIP_TEST=false
SKIP_RESULTS=false
SKIP_COMPARE=false
MODE_VALUE=""
CONFIG_VALUE=""
MODEL_DIR_VALUE=""

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE_VALUE="${arg#*=}"
            shift
            ;;
        --config=*|--prueba=*)
            CONFIG_VALUE="${arg#*=}"
            shift
            ;;
        --model-dir=*)
            MODEL_DIR_VALUE="${arg#*=}"
            shift
            ;;
        -strain|--skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        -stest|--skip-test)
            SKIP_TEST=true
            shift
            ;;
        -sres|--skip-results)
            SKIP_RESULTS=true
            shift
            ;;
        -scomp|--skip-compare)
            SKIP_COMPARE=true
            shift
            ;;
        *)
            ;;
    esac
done

# Apply configuration overrides to main_config.ini
if [ -n "$MODE_VALUE" ]; then
    sed -i '' "s/^mode *= *.*/mode = ${MODE_VALUE}/" main_config.ini 2>/dev/null || sed -i "s/^mode *= *.*/mode = ${MODE_VALUE}/" main_config.ini
fi

if [ -n "$CONFIG_VALUE" ]; then
    sed -i '' "s/^prueba *= *.*/prueba = ${CONFIG_VALUE}/" main_config.ini 2>/dev/null || sed -i "s/^prueba *= *.*/prueba = ${CONFIG_VALUE}/" main_config.ini
fi

if [ -n "$MODEL_DIR_VALUE" ]; then
    sed -i '' "s|^model_dir *= *.*|model_dir = ${MODEL_DIR_VALUE}|" main_config.ini 2>/dev/null || sed -i "s|^model_dir *= *.*|model_dir = ${MODEL_DIR_VALUE}|" main_config.ini
fi

# Read updated values from main_config.ini
MODE=$($PYTHON_BIN -c "import configparser; c=configparser.ConfigParser(); c.read('main_config.ini'); print(c['CONFIG'].get('mode', 'direct'))")
PRUEBA=$($PYTHON_BIN -c "import configparser; c=configparser.ConfigParser(); c.read('main_config.ini'); print(c['CONFIG'].get('prueba', 'alexnet_original'))")
MAIN_PATH=$($PYTHON_BIN -c "import configparser; c=configparser.ConfigParser(); c.read('main_config.ini'); print(c['PATHS'].get('main_path', './csst_catalog/test25_tesis/'))")
MODEL_DIR=$($PYTHON_BIN -c "import configparser; c=configparser.ConfigParser(); c.read('main_config.ini'); print(c['CONFIG'].get('model_dir', ''))")

LOG_DIR="${MAIN_PATH}/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +"%Y_%m_%d_%H_%M_%S")
LOG_FILE="$LOG_DIR/analysis_${PRUEBA}_$DATE.log"

echo "${YELLOW}================================================================${NC}" | tee -a "$LOG_FILE"
echo "${YELLOW} Starting Lensing Analysis Pipeline at $(date)${NC}" | tee -a "$LOG_FILE"
echo "${CYAN} Mode      : ${MODE}${NC}" | tee -a "$LOG_FILE"
echo "${CYAN} Prueba    : ${PRUEBA}${NC}" | tee -a "$LOG_FILE"
if [ -n "$MODEL_DIR" ]; then
    echo "${CYAN} Model Dir : ${MODEL_DIR}${NC}" | tee -a "$LOG_FILE"
fi
echo "${YELLOW}================================================================${NC}" | tee -a "$LOG_FILE"

# Step 1: Model Training
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n${CYAN}[1/4] Running Model Training...${NC}" | tee -a "$LOG_FILE"
    if [ -f "alexnet_test.py" ]; then
        $PYTHON_BIN alexnet_test.py >> "$LOG_FILE" 2>&1
        if [ $? -ne 0 ]; then
            echo "${RED}Error during training. Check log file: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
            exit 1
        else
            echo "${GREEN}Training completed successfully.${NC}" | tee -a "$LOG_FILE"
        fi
    else
        echo "${YELLOW}No training script found. Skipping training.${NC}" | tee -a "$LOG_FILE"
    fi
else
    echo -e "\n[1/4] Skipping model training." | tee -a "$LOG_FILE"
fi

# Step 2: Model Testing
if [ "$SKIP_TEST" = false ]; then
    if [ "$MODE" = "deepensemble" ]; then
        echo -e "\n${CYAN}[2/4] Running Deep Ensemble Testing (test_deepensamble.py)...${NC}" | tee -a "$LOG_FILE"
        $PYTHON_BIN test_deepensamble.py >> "$LOG_FILE" 2>&1
    else
        echo -e "\n${CYAN}[2/4] Running Direct Model Testing (test.py)...${NC}" | tee -a "$LOG_FILE"
        $PYTHON_BIN test.py >> "$LOG_FILE" 2>&1
    fi
    if [ $? -ne 0 ]; then
        echo "${RED}Error during model testing. Check log file: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
        exit 1
    else
        echo "${GREEN}Model testing completed successfully.${NC}" | tee -a "$LOG_FILE"
    fi
else
    echo -e "\n[2/4] Skipping model testing." | tee -a "$LOG_FILE"
fi

# Step 3: Error & Results Calculation
if [ "$SKIP_RESULTS" = false ]; then
    echo -e "\n${CYAN}[3/4] Running Error & Results Calculation (results_paper.py)...${NC}" | tee -a "$LOG_FILE"
    $PYTHON_BIN results_paper.py >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "${RED}Error during results calculation. Check log file: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
        exit 1
    else
        echo "${GREEN}Results calculation completed successfully.${NC}" | tee -a "$LOG_FILE"
    fi
else
    echo -e "\n[3/4] Skipping results calculation." | tee -a "$LOG_FILE"
fi

# Step 4: Image Comparison (PSNR & SSIM)
if [ "$SKIP_COMPARE" = false ]; then
    echo -e "\n${CYAN}[4/4] Running Image Reconstruction Comparison (compare_images.py & psnr_total.py)...${NC}" | tee -a "$LOG_FILE"
    
    echo "  > Single-image PSNR comparison (compare_images.py)..." | tee -a "$LOG_FILE"
    $PYTHON_BIN compare_images.py >> "$LOG_FILE" 2>&1
    
    echo "  > Dataset-wide SSIM (Global & Einstein Ring) & PSNR evaluation (psnr_total.py)..." | tee -a "$LOG_FILE"
    $PYTHON_BIN psnr_total.py >> "$LOG_FILE" 2>&1
    
    if [ "$MODE" = "deepensemble" ] && [ -f "psnr_deepensemble.py" ]; then
        echo "  > Ensemble PSNR comparison (psnr_deepensemble.py)..." | tee -a "$LOG_FILE"
        $PYTHON_BIN psnr_deepensemble.py >> "$LOG_FILE" 2>&1
    fi

    if [ $? -ne 0 ]; then
        echo "${RED}Error during image comparison evaluation. Check log file: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
        exit 1
    else
        echo "${GREEN}Image comparison & SSIM/PSNR evaluation completed successfully.${NC}" | tee -a "$LOG_FILE"
    fi
else
    echo -e "\n[4/4] Skipping image comparison & SSIM evaluation." | tee -a "$LOG_FILE"
fi

echo -e "\n${YELLOW}================================================================${NC}" | tee -a "$LOG_FILE"
echo "${YELLOW} Pipeline execution completed at $(date)${NC}" | tee -a "$LOG_FILE"
echo "${YELLOW} Log file saved to: $LOG_FILE${NC}"
echo "${YELLOW}================================================================${NC}"
