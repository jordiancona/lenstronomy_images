#!/usr/bin/env bash
set -euo pipefail

###################################################################
# Script para gestionar Datasets y Entrenamientos en Kaggle
###################################################################

# Colores para la terminal
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
RED=$'\033[0;31m'
BLUE=$'\033[0;34m'
BOLD=$'\033[1m'
NC=$'\033[0m'

# Valores por defecto
MODE=""                  # dataset | kernel / script
KERNEL_ID=""             # ID del kernel en Kaggle
PATH_DIR=""              # Directorio objetivo
POLL_INTERVAL=30         # Tiempo de espera entre consultas (segundos)
OUTPUT_DIR="./output"    # Directorio para guardar outputs
NON_INTERACTIVE=false    # Modo no interactivo

# Helper para ejecutar Kaggle CLI reconociendo entornos pyenv o python
run_kaggle() {
    if PYENV_VERSION=project kaggle --version &>/dev/null; then
        PYENV_VERSION=project kaggle "$@"
    elif kaggle --version &>/dev/null; then
        kaggle "$@"
    elif python3 -m kaggle --version &>/dev/null; then
        python3 -m kaggle "$@"
    else
        echo "${RED}Error: La herramienta CLI 'kaggle' no está instalada o no se encuentra en el PATH.${NC}" >&2
        exit 1
    fi
}

# Helper para obtener un campo de un archivo JSON
get_json_field() {
    local json_file="$1"
    local field="$2"
    if [ -f "$json_file" ]; then
        python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
        val = data.get(sys.argv[2], "")
        if isinstance(val, list):
            print(", ".join(str(x) for x in val))
        elif isinstance(val, bool):
            print("true" if val else "false")
        else:
            print(val if val is not None else "")
except Exception:
    print("")
' "$json_file" "$field"
    else
        echo ""
    fi
}

# Helper para determinar el directorio objetivo por defecto
determine_default_path_dir() {
    local check_file="$1" # dataset-metadata.json o kernel-metadata.json
    if [ -f "./$check_file" ]; then
        echo "."
    elif [ -d "./kaggle_runs" ]; then
        echo "./kaggle_runs"
    else
        echo "."
    fi
}

# Helper para solicitar entrada interactiva con valor por defecto
prompt_input() {
    local prompt_msg="$1"
    local default_val="$2"
    local var_name="$3"

    if [ -n "$default_val" ]; then
        echo -n -e "${CYAN}${prompt_msg}${NC} [${YELLOW}${default_val}${NC}]: " >&2
    else
        echo -n -e "${CYAN}${prompt_msg}${NC}: " >&2
    fi

    read -r input_val
    if [ -z "$input_val" ]; then
        eval "$var_name=\"\$default_val\""
    else
        eval "$var_name=\"\$input_val\""
    fi
}

usage() {
    cat << EOF
${BOLD}Uso:${NC} $0 [opciones]

${BOLD}Opciones:${NC}
  ${BLUE}-m, --mode <dataset|kernel>${NC}  Modo de operación (dataset o kernel)
  ${BLUE}-k, --kernel-id <id>${NC}        ID del kernel en Kaggle (ej: usuario/mi-modelo)
  ${BLUE}-p, --path-dir <ruta>${NC}         Directorio local que contiene el código o datos
  ${BLUE}-o, --output-dir <ruta>${NC}       Directorio para guardar artefactos descargados (default: ./output)
  ${BLUE}-i, --interval <segundos>${NC}     Intervalo de consulta de estado del kernel (default: 30)
  ${BLUE}-y, --yes${NC}                    Modo no interactivo (usa valores existentes/por defecto)
  ${BLUE}-h, --help${NC}                   Muestra esta ayuda
EOF
}

# Procesar argumentos de línea de comandos
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -k|--kernel-id)
            KERNEL_ID="$2"
            shift 2
            ;;
        -p|--path-dir)
            PATH_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -i|--interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        -y|--yes|--non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # Soporte posicional opcional
            if [ -z "$MODE" ] && [[ "$1" == "dataset" || "$1" == "kernel" || "$1" == "script" ]]; then
                MODE="$1"
            elif [ -z "$KERNEL_ID" ]; then
                KERNEL_ID="$1"
            elif [ -z "$PATH_DIR" ]; then
                PATH_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Normalizar alias del modo
if [[ "$MODE" == "script" ]]; then
    MODE="kernel"
fi

# Expandir tilde en PATH_DIR si fue proporcionado por CLI
if [[ "$PATH_DIR" == "~"* ]]; then
    PATH_DIR="${PATH_DIR/#\~/$HOME}"
fi

# Flujo interactivo inicial si no se especificó el modo
if [ -z "$MODE" ] && [ "$NON_INTERACTIVE" = false ]; then
    echo "${BOLD}${BLUE}====================================================${NC}"
    echo "${BOLD}${BLUE}      Gestor de Kaggle: Datasets y Kernels          ${NC}"
    echo "${BOLD}${BLUE}====================================================${NC}"
    echo "Selecciona la operación que deseas realizar:"
    echo "  ${GREEN}1)${NC} Subir una base de datos (Dataset)"
    echo "  ${GREEN}2)${NC} Subir y ejecutar un script/notebook (Kernel)"
    echo ""
    
    prompt_input "Selecciona una opción [1-2]" "2" OPTION
    case "$OPTION" in
        1) MODE="dataset" ;;
        2|*) MODE="kernel" ;;
    esac
fi

# Si aún no se define el modo, por defecto es kernel
if [ -z "$MODE" ]; then
    MODE="kernel"
fi

###################################################################
# FLUJO 1: SUBIR BASE DE DATOS
###################################################################
if [ "$MODE" = "dataset" ]; then
    echo ""
    echo "${YELLOW}=== Configuración de Subida de Dataset a Kaggle ===${NC}"

    if [ -z "$PATH_DIR" ]; then
        DEFAULT_PATH=$(determine_default_path_dir "dataset-metadata.json")
        if [ "$NON_INTERACTIVE" = false ]; then
            prompt_input "Directorio con los datos a subir" "$DEFAULT_PATH" PATH_DIR
        else
            PATH_DIR="$DEFAULT_PATH"
        fi
    fi

    if [[ "$PATH_DIR" == "~"* ]]; then
        PATH_DIR="${PATH_DIR/#\~/$HOME}"
    fi

    if [ ! -d "$PATH_DIR" ]; then
        echo "${RED}Error: El directorio '$PATH_DIR' no existe.${NC}"
        exit 1
    fi

    META_FILE="$PATH_DIR/dataset-metadata.json"
    
    # Leer valores existentes si el archivo ya existe
    DEFAULT_TITLE=$(get_json_field "$META_FILE" "title")
    DEFAULT_ID=$(get_json_field "$META_FILE" "id")
    DEFAULT_LICENSE="CC0-1.0"

    [ -z "$DEFAULT_TITLE" ] && DEFAULT_TITLE="Mi Dataset de Kaggle"
    [ -z "$DEFAULT_ID" ] && DEFAULT_ID="usuario/mi-dataset"

    if [ "$NON_INTERACTIVE" = false ]; then
        echo ""
        echo "${CYAN}Ingresa o confirma los valores para dataset-metadata.json:${NC}"
        prompt_input "ID del Dataset (usuario/slug)" "$DEFAULT_ID" DS_ID
        prompt_input "Título del Dataset" "$DEFAULT_TITLE" DS_TITLE
        prompt_input "Licencia" "$DEFAULT_LICENSE" DS_LICENSE
    else
        DS_ID="$DEFAULT_ID"
        DS_TITLE="$DEFAULT_TITLE"
        DS_LICENSE="$DEFAULT_LICENSE"
    fi

    # Generar/actualizar dataset-metadata.json mediante Python
    python3 -c '
import json, sys
filepath, ds_id, title, license_name = sys.argv[1:5]
metadata = {
    "title": title,
    "id": ds_id,
    "licenses": [{"name": license_name}]
}
with open(filepath, "w") as f:
    json.dump(metadata, f, indent=2)
' "$META_FILE" "$DS_ID" "$DS_TITLE" "$DS_LICENSE"

    echo "${GREEN}✓ 'dataset-metadata.json' actualizado en '$PATH_DIR'.${NC}"
    echo ""

    # Determinar si es un dataset nuevo o una nueva versión
    SUBIDA_TIPO="1"
    if [ "$NON_INTERACTIVE" = false ]; then
        echo "Selecciona el tipo de subida:"
        echo "  ${GREEN}1)${NC} Crear un NUEVO dataset"
        echo "  ${GREEN}2)${NC} Crear una NUEVA VERSIÓN de un dataset existente"
        prompt_input "Opción [1-2]" "1" SUBIDA_TIPO
    fi

    if [ "$SUBIDA_TIPO" = "2" ]; then
        VERSION_MSG="Actualización de datos"
        if [ "$NON_INTERACTIVE" = false ]; then
            prompt_input "Mensaje o notas de la nueva versión" "$VERSION_MSG" VERSION_MSG
        fi
        echo "${YELLOW}=== Creando nueva versión del dataset en Kaggle ===${NC}"
        run_kaggle datasets version -p "$PATH_DIR" -m "$VERSION_MSG"
    else
        IS_PUBLIC="n"
        if [ "$NON_INTERACTIVE" = false ]; then
            prompt_input "¿Hacer el dataset público? (s/N)" "n" IS_PUBLIC
        fi
        
        echo "${YELLOW}=== Creando nuevo dataset en Kaggle ===${NC}"
        if [[ "$IS_PUBLIC" =~ ^[sSyY] ]]; then
            run_kaggle datasets create -p "$PATH_DIR" -u
        else
            run_kaggle datasets create -p "$PATH_DIR"
        fi
    fi

    echo "${GREEN}=== Proceso de carga de dataset finalizado ===${NC}"
    exit 0
fi

###################################################################
# FLUJO 2: SUBIR Y EJECUTAR SCRIPT / KERNEL
###################################################################
echo ""
echo "${YELLOW}=== Configuración de Script / Kernel para Kaggle ===${NC}"

if [ -z "$PATH_DIR" ]; then
    DEFAULT_PATH=$(determine_default_path_dir "kernel-metadata.json")
    if [ "$NON_INTERACTIVE" = false ]; then
        prompt_input "Directorio con el código del kernel" "$DEFAULT_PATH" PATH_DIR
    else
        PATH_DIR="$DEFAULT_PATH"
    fi
fi

if [[ "$PATH_DIR" == "~"* ]]; then
    PATH_DIR="${PATH_DIR/#\~/$HOME}"
fi

if [ ! -d "$PATH_DIR" ]; then
    echo "${RED}Error: El directorio '$PATH_DIR' no existe.${NC}"
    exit 1
fi

META_FILE="$PATH_DIR/kernel-metadata.json"

# Cargar valores por defecto desde el metadata actual si existe
DEF_ID=$(get_json_field "$META_FILE" "id")
DEF_TITLE=$(get_json_field "$META_FILE" "title")
DEF_CODE=$(get_json_field "$META_FILE" "code_file")
DEF_LANG=$(get_json_field "$META_FILE" "language")
DEF_TYPE=$(get_json_field "$META_FILE" "kernel_type")
DEF_PRIV=$(get_json_field "$META_FILE" "is_private")
DEF_GPU=$(get_json_field "$META_FILE" "enable_gpu")
DEF_TPU=$(get_json_field "$META_FILE" "enable_tpu")
DEF_NET=$(get_json_field "$META_FILE" "enable_internet")
DEF_DATASETS=$(get_json_field "$META_FILE" "dataset_sources")
DEF_COMPS=$(get_json_field "$META_FILE" "competition_sources")
DEF_KERNELS=$(get_json_field "$META_FILE" "kernel_sources")
DEF_MODELS=$(get_json_field "$META_FILE" "model_sources")

# Asignar valores fallback si estaba vacío
[ -z "$DEF_ID" ] && DEF_ID="jordiancona/lenses-training"
[ -z "$DEF_TITLE" ] && DEF_TITLE="Lenses training"
[ -z "$DEF_LANG" ] && DEF_LANG="python"
[ -z "$DEF_TYPE" ] && DEF_TYPE="notebook"
[ -z "$DEF_PRIV" ] && DEF_PRIV="true"
[ -z "$DEF_GPU" ] && DEF_GPU="true"
[ -z "$DEF_TPU" ] && DEF_TPU="false"
[ -z "$DEF_NET" ] && DEF_NET="true"

# Si no había code_file especificado, buscar archivos en el directorio
if [ -z "$DEF_CODE" ]; then
    FOUND_FILES=($(find "$PATH_DIR" -maxdepth 1 \( -name "*.ipynb" -o -name "*.py" \) -exec basename {} \; 2>/dev/null || true))
    if [ ${#FOUND_FILES[@]} -gt 0 ]; then
        DEF_CODE="${FOUND_FILES[0]}"
    else
        DEF_CODE="models-test-lenses.ipynb"
    fi
fi

# Si se pasó KERNEL_ID por argumento CLI, anula el por defecto
if [ -n "$KERNEL_ID" ]; then
    DEF_ID="$KERNEL_ID"
fi

if [ "$NON_INTERACTIVE" = false ]; then
    echo ""
    echo "${CYAN}Por favor ingresa o confirma los valores para kernel-metadata.json:${NC}"

    prompt_input "ID del Kernel (usuario/slug)" "$DEF_ID" K_ID
    prompt_input "Título del Kernel" "$DEF_TITLE" K_TITLE

    # Selección de archivo de código
    FILES=($(find "$PATH_DIR" -maxdepth 1 \( -name "*.ipynb" -o -name "*.py" \) -exec basename {} \; 2>/dev/null || true))
    if [ ${#FILES[@]} -gt 0 ]; then
        echo ""
        echo "${CYAN}Archivos de código detectados en '$PATH_DIR':${NC}"
        for idx in "${!FILES[@]}"; do
            echo "  ${GREEN}$((idx+1)))${NC} ${FILES[$idx]}"
        done
        prompt_input "Número o nombre del archivo de código" "$DEF_CODE" CODE_CHOICE

        if [[ "$CODE_CHOICE" =~ ^[0-9]+$ ]] && [ "$CODE_CHOICE" -ge 1 ] && [ "$CODE_CHOICE" -le "${#FILES[@]}" ]; then
            K_CODE="${FILES[$((CODE_CHOICE-1))]}"
        else
            K_CODE="$CODE_CHOICE"
        fi
    else
        prompt_input "Nombre del archivo de código" "$DEF_CODE" K_CODE
    fi

    # Determinar tipo según extensión si no fue cambiado
    if [[ "$K_CODE" == *.py ]]; then
        DEF_TYPE="script"
    elif [[ "$K_CODE" == *.ipynb ]]; then
        DEF_TYPE="notebook"
    fi

    prompt_input "Lenguaje" "$DEF_LANG" K_LANG
    prompt_input "Tipo de Kernel (notebook/script)" "$DEF_TYPE" K_TYPE
    prompt_input "¿Es Privado? (true/false)" "$DEF_PRIV" K_PRIV
    prompt_input "¿Habilitar GPU? (true/false)" "$DEF_GPU" K_GPU
    prompt_input "¿Habilitar TPU? (true/false)" "$DEF_TPU" K_TPU
    prompt_input "¿Habilitar Internet? (true/false)" "$DEF_NET" K_NET
    prompt_input "Fuentes de Dataset (separadas por coma)" "$DEF_DATASETS" K_DATASETS
    prompt_input "Fuentes de Competencias (separadas por coma)" "$DEF_COMPS" K_COMPS
    prompt_input "Fuentes de Kernels (separadas por coma)" "$DEF_KERNELS" K_KERNELS
    prompt_input "Fuentes de Modelos (separadas por coma)" "$DEF_MODELS" K_MODELS
else
    K_ID="$DEF_ID"
    K_TITLE="$DEF_TITLE"
    K_CODE="$DEF_CODE"
    K_LANG="$DEF_LANG"
    K_TYPE="$DEF_TYPE"
    K_PRIV="$DEF_PRIV"
    K_GPU="$DEF_GPU"
    K_TPU="$DEF_TPU"
    K_NET="$DEF_NET"
    K_DATASETS="$DEF_DATASETS"
    K_COMPS="$DEF_COMPS"
    K_KERNELS="$DEF_KERNELS"
    K_MODELS="$DEF_MODELS"
fi

KERNEL_ID="$K_ID"

# Guardar/actualizar kernel-metadata.json con Python
python3 -c '
import json, sys

filepath, kid, title, code_file, lang, ktype, priv, gpu, tpu, net, dsets, comps, kerns, mdls = sys.argv[1:15]

def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_bool(b):
    return "true" if b.lower() in ("true", "1", "yes", "s", "y") else "false"

metadata = {
    "id": kid,
    "title": title,
    "code_file": code_file,
    "language": lang,
    "kernel_type": ktype,
    "is_private": parse_bool(priv),
    "enable_gpu": parse_bool(gpu),
    "enable_tpu": parse_bool(tpu),
    "enable_internet": parse_bool(net),
    "machine_shape": "",
    "dataset_sources": parse_list(dsets),
    "competition_sources": parse_list(comps),
    "kernel_sources": parse_list(kerns),
    "model_sources": parse_list(mdls)
}

with open(filepath, "w") as f:
    json.dump(metadata, f, indent=4)
' "$META_FILE" "$KERNEL_ID" "$K_TITLE" "$K_CODE" "$K_LANG" "$K_TYPE" "$K_PRIV" "$K_GPU" "$K_TPU" "$K_NET" "$K_DATASETS" "$K_COMPS" "$K_KERNELS" "$K_MODELS"

echo ""
echo "${GREEN}✓ 'kernel-metadata.json' configurado exitosamente:${NC}"
echo "  - ID: $KERNEL_ID"
echo "  - Código: $K_CODE"
echo "  - GPU: $K_GPU | TPU: $K_TPU | Internet: $K_NET"
echo ""

# Confirmación de envío
RUN_NOW="s"
if [ "$NON_INTERACTIVE" = false ]; then
    prompt_input "¿Deseas subir y ejecutar este kernel en Kaggle ahora? (S/n)" "s" RUN_NOW
fi

if [[ ! "$RUN_NOW" =~ ^[sSyY] ]]; then
    echo "${YELLOW}Configuración guardada en '$META_FILE'. Ejecución cancelada por el usuario.${NC}"
    exit 0
fi

# Normalizar KERNEL_ID (Kaggle convierte '_' a '-' en el slug)
if [[ "$KERNEL_ID" == *"_"* ]]; then
    KERNEL_ID="${KERNEL_ID//_/-}"
fi

# Ejecución en Kaggle
echo "${YELLOW}=== Enviando Kernel: $KERNEL_ID ===${NC}"
run_kaggle kernels push -p "$PATH_DIR"

echo "${YELLOW}=== Monitoreando ejecución (actualización cada ${POLL_INTERVAL}s) ===${NC}"
while true; do
    STATUS_OUTPUT=$(run_kaggle kernels status "$KERNEL_ID" 2>&1 || true)

    # Si kaggle falla por slug con guiones bajos (ej: usuario/kernel_slug vs usuario/kernel-slug), probar con '-'
    if echo "$STATUS_OUTPUT" | grep -qiE "(wrong kernel slug|permission.*denied|cannot access)"; then
        ALT_KERNEL_ID="${KERNEL_ID//_/-}"
        if [ "$ALT_KERNEL_ID" != "$KERNEL_ID" ]; then
            ALT_OUTPUT=$(run_kaggle kernels status "$ALT_KERNEL_ID" 2>&1 || true)
            if ! echo "$ALT_OUTPUT" | grep -qiE "(wrong kernel slug|permission.*denied|cannot access)"; then
                echo "${YELLOW}Aviso: Ajustando KERNEL_ID a '$ALT_KERNEL_ID' (Kaggle convierte '_' a '-').${NC}"
                KERNEL_ID="$ALT_KERNEL_ID"
                STATUS_OUTPUT="$ALT_OUTPUT"
            fi
        fi
    fi

    RAW_STATUS=""
    if [[ "$STATUS_OUTPUT" =~ has[[:space:]]+status[[:space:]]+\"([^\"]+)\" ]]; then
        RAW_STATUS="${BASH_REMATCH[1]}"
    elif [[ "$STATUS_OUTPUT" =~ has[[:space:]]+status[[:space:]]+([^[:space:]]+) ]]; then
        RAW_STATUS="${BASH_REMATCH[1]}"
    fi

    # Limpiar prefijos y convertir a minúsculas
    RAW_STATUS="${RAW_STATUS#KernelWorkerStatus.}"
    CLEAN_STATUS=$(echo "$RAW_STATUS" | tr '[:upper:]' '[:lower:]')

    case "$CLEAN_STATUS" in
        queued)
            STATUS="queued"
            ;;
        running)
            STATUS="running"
            ;;
        complete)
            STATUS="complete"
            ;;
        error|failed)
            STATUS="error"
            ;;
        cancelack|cancel_ack|canceled|cancelled)
            STATUS="cancelAck"
            ;;
        canceling|cancelling)
            STATUS="canceling"
            ;;
        *)
            if [ -n "$CLEAN_STATUS" ]; then
                STATUS="$CLEAN_STATUS"
            else
                STATUS="unknown"
            fi
            ;;
    esac

    if [ "$STATUS" = "unknown" ]; then
        FIRST_LINE=$(echo "$STATUS_OUTPUT" | head -n 1)
        echo "[$(date +'%H:%M:%S')] ${YELLOW}Estado: unknown (${FIRST_LINE})${NC}"
    else
        echo "[$(date +'%H:%M:%S')] ${GREEN}Estado: $STATUS${NC}"
    fi

    case "$STATUS" in
        "complete")
            echo "${GREEN}-> ¡Ejecución terminada exitosamente!${NC}"
            break
            ;;
        "error")
            echo "${RED}-> Falló la ejecución del kernel en Kaggle.${NC}"

            echo "${YELLOW}---- Detalle del error ----${NC}"
            TMP_LOG_DIR=$(mktemp -d)
            if run_kaggle kernels output "$KERNEL_ID" -p "$TMP_LOG_DIR" >/dev/null 2>&1; then
                LOG_FILE=$(ls "$TMP_LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)
                if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
                    python3 -c '
import json, sys
log_file = sys.argv[1]
try:
    with open(log_file, "r") as f:
        data = json.load(f)
        for item in data:
            if isinstance(item, dict) and "data" in item:
                sys.stdout.write(item["data"])
except Exception:
    with open(log_file, "r") as f:
        print(f.read())
' "$LOG_FILE" 2>/dev/null || cat "$LOG_FILE"
                else
                    echo "No se encontró archivo de log en la salida del kernel."
                fi
            else
                echo "No se pudo obtener el detalle de salida de Kaggle."
            fi
            rm -rf "$TMP_LOG_DIR"
            exit 1
            ;;
        "cancelAck")
            echo "${RED}-> La ejecución fue cancelada.${NC}"
            exit 1
            ;;
        *)
            sleep "$POLL_INTERVAL"
            ;;
    esac
done

# Descarga de resultados
echo "${YELLOW}=== Descargando salidas a '$OUTPUT_DIR' ===${NC}"
mkdir -p "$OUTPUT_DIR"
run_kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"

echo "${BOLD}${GREEN}=== Proceso finalizado ===${NC}"

