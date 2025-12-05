#!/usr/bin/env bash

# Experiment Configuration Templates
# ==================================
# Ready-to-use commands for common experimental scenarios

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="/Users/theodora/Desktop/uni/project/Neural-LSH"

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# ============================================================================
# SCENARIO 1: QUICK TEST (5-10 minutes)
# ============================================================================
scenario_quick_test() {
    print_header "Quick Test: Verify Framework Works"
    echo "Configurations: 20"
    echo "Estimated Time: 10-30 minutes"
    echo "Best for: Testing setup and getting quick results\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 10 15 \
        --m 50 100 \
        --layers 2 \
        --nodes 32 64 \
        --epochs 25 \
        --data sift
}

# ============================================================================
# SCENARIO 2: BALANCED STUDY (3-6 hours)
# ============================================================================
scenario_balanced() {
    print_header "Balanced Study: Good Coverage Without Excessive Time"
    echo "Configurations: 60"
    echo "Estimated Time: 3-6 hours"
    echo "Best for: Comprehensive analysis with reasonable runtime\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 10 15 20 \
        --m 50 100 200 \
        --layers 2 3 \
        --nodes 32 64 128 \
        --epochs 25 50 \
        --data sift
}

# ============================================================================
# SCENARIO 3: FULL STUDY (8-16 hours)
# ============================================================================
scenario_full() {
    print_header "Full Study: Complete Hyperparameter Space"
    echo "Configurations: 162"
    echo "Estimated Time: 8-16 hours"
    echo "Best for: Most comprehensive analysis\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 10 15 20 \
        --m 50 100 200 \
        --layers 2 3 4 \
        --nodes 32 64 128 \
        --epochs 25 50 \
        --data sift
}

# ============================================================================
# SCENARIO 4: FOCUSED GRID (1-2 hours)
# ============================================================================
scenario_focused() {
    print_header "Focused Grid: Depth in Few Dimensions"
    echo "Focus: Optimize m (partitions) and layers"
    echo "Configurations: 30"
    echo "Estimated Time: 1-2 hours\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 15 \
        --m 50 100 150 200 250 \
        --layers 2 3 4 5 \
        --nodes 64 \
        --epochs 50 \
        --data sift
}

# ============================================================================
# SCENARIO 5: MNIST STUDY (2-4 hours)
# ============================================================================
scenario_mnist() {
    print_header "MNIST Study: Smaller Dataset"
    echo "Configurations: 60"
    echo "Estimated Time: 2-4 hours"
    echo "Best for: Faster experiments on smaller dataset\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 5 10 15 \
        --m 50 100 200 \
        --layers 2 3 4 \
        --nodes 32 64 \
        --epochs 25 50 \
        --data mnist
}

# ============================================================================
# SCENARIO 6: COMPARE REPORT GENERATION (< 1 minute)
# ============================================================================
scenario_reports_only() {
    print_header "Report Generation Only"
    echo "Regenerates all reports from existing results"
    echo "Estimated Time: < 1 minute\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py --skip-grid-search
}

# ============================================================================
# SCENARIO 7: ARCHITECTURE SEARCH (2-4 hours)
# ============================================================================
scenario_architecture() {
    print_header "MLP Architecture Search"
    echo "Focus: Optimize MLP layers and nodes"
    echo "Configurations: 36"
    echo "Estimated Time: 2-4 hours\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 15 \
        --m 100 \
        --layers 1 2 3 4 5 6 \
        --nodes 16 32 64 128 \
        --epochs 50 \
        --data sift
}

# ============================================================================
# SCENARIO 8: HYPERPARAMETER SENSITIVITY (1-2 hours)
# ============================================================================
scenario_sensitivity() {
    print_header "Hyperparameter Sensitivity Analysis"
    echo "Focus: k (graph parameter)"
    echo "Configurations: 18"
    echo "Estimated Time: 1-2 hours\n"
    
    cd "$PROJECT_DIR"
    python experimental_study/run_experiments.py \
        --knn 5 10 15 20 25 30 \
        --m 100 \
        --layers 3 \
        --nodes 64 \
        --epochs 50 \
        --data sift
}

# ============================================================================
# SCENARIO 9: QUICK COMPARATIVE ANALYSIS (< 1 minute)
# ============================================================================
scenario_comparison() {
    print_header "Comparative Analysis Only"
    echo "Compares best Neural-LSH config against old methods"
    echo "Estimated Time: < 1 minute\n"
    
    cd "$PROJECT_DIR"
    
    echo -e "${YELLOW}SIFT Dataset Comparison:${NC}"
    python experimental_study/comparative_analysis.py --dataset sift
    
    echo -e "\n${YELLOW}MNIST Dataset Comparison:${NC}"
    python experimental_study/comparative_analysis.py --dataset mnist
}

# ============================================================================
# SCENARIO 10: CUSTOM (User Defined)
# ============================================================================
scenario_custom() {
    print_header "Custom Experiment Configuration"
    echo "Enter your own hyperparameter values"
    echo ""
    
    read -p "k-NN values (space-separated, default: 10 15): " knn
    knn=${knn:-"10 15"}
    
    read -p "Partitions (space-separated, default: 50 100): " m
    m=${m:-"50 100"}
    
    read -p "MLP layers (space-separated, default: 2 3): " layers
    layers=${layers:-"2 3"}
    
    read -p "MLP nodes (space-separated, default: 32 64): " nodes
    nodes=${nodes:-"32 64"}
    
    read -p "Epochs (space-separated, default: 25 50): " epochs
    epochs=${epochs:-"25 50"}
    
    read -p "Dataset (sift/mnist, default: sift): " data
    data=${data:-"sift"}
    
    read -p "Max experiments (leave empty for all): " max_exp
    
    cd "$PROJECT_DIR"
    
    cmd="python experimental_study/run_experiments.py"
    cmd="$cmd --knn $knn"
    cmd="$cmd --m $m"
    cmd="$cmd --layers $layers"
    cmd="$cmd --nodes $nodes"
    cmd="$cmd --epochs $epochs"
    cmd="$cmd --data $data"
    
    if [ -n "$max_exp" ]; then
        cmd="$cmd --max-experiments $max_exp"
    fi
    
    echo -e "\n${GREEN}Running:${NC} $cmd\n"
    eval $cmd
}

# ============================================================================
# MAIN MENU
# ============================================================================
show_menu() {
    echo ""
    echo -e "${GREEN}Neural-LSH Experimental Study - Scenario Menu${NC}"
    echo "=============================================="
    echo ""
    echo "1) Quick Test (5-10 min, verify setup)"
    echo "2) Balanced Study (3-6 hours, good coverage)"
    echo "3) Full Study (8-16 hours, complete grid)"
    echo "4) Focused Grid (1-2 hours, optimize partitions & layers)"
    echo "5) MNIST Study (2-4 hours, smaller dataset)"
    echo "6) Reports Only (< 1 min, regenerate from results)"
    echo "7) Architecture Search (2-4 hours, optimize MLP)"
    echo "8) Sensitivity Analysis (1-2 hours, vary k)"
    echo "9) Comparative Analysis (< 1 min, vs old methods)"
    echo "10) Custom Configuration (user defined)"
    echo "0) Exit"
    echo ""
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================
main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select scenario (0-10): " choice
            
            case $choice in
                1) scenario_quick_test ;;
                2) scenario_balanced ;;
                3) scenario_full ;;
                4) scenario_focused ;;
                5) scenario_mnist ;;
                6) scenario_reports_only ;;
                7) scenario_architecture ;;
                8) scenario_sensitivity ;;
                9) scenario_comparison ;;
                10) scenario_custom ;;
                0) echo "Exiting..."; exit 0 ;;
                *) echo "Invalid choice. Please select 0-10." ;;
            esac
            
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            quick) scenario_quick_test ;;
            balanced) scenario_balanced ;;
            full) scenario_full ;;
            focused) scenario_focused ;;
            mnist) scenario_mnist ;;
            reports) scenario_reports_only ;;
            architecture) scenario_architecture ;;
            sensitivity) scenario_sensitivity ;;
            comparison) scenario_comparison ;;
            custom) scenario_custom ;;
            *)
                echo "Usage: ./run_experiment_scenario.sh [scenario]"
                echo ""
                echo "Scenarios:"
                echo "  quick          - Quick test (5-10 min)"
                echo "  balanced       - Balanced study (3-6 hours)"
                echo "  full           - Full study (8-16 hours)"
                echo "  focused        - Focused grid (1-2 hours)"
                echo "  mnist          - MNIST study (2-4 hours)"
                echo "  reports        - Reports only (< 1 min)"
                echo "  architecture   - Architecture search (2-4 hours)"
                echo "  sensitivity    - Sensitivity analysis (1-2 hours)"
                echo "  comparison     - Comparative analysis (< 1 min)"
                echo "  custom         - Custom configuration"
                echo ""
                echo "Examples:"
                echo "  ./run_experiment_scenario.sh quick"
                echo "  ./run_experiment_scenario.sh balanced"
                echo "  ./run_experiment_scenario.sh full"
                exit 1
                ;;
        esac
    fi
}

# Run main function
main "$@"
