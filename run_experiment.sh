#!/bin/bash

# ESG RAG Experiment Runner
# This script runs the complete 3-step evaluation pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_error "Virtual environment is not activated!"
        echo "Please run: source venv/bin/activate"
        exit 1
    fi
    print_success "Virtual environment is activated"
}

# Function to check if Ollama is running
check_ollama() {
    if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        print_error "Ollama is not running!"
        echo "Please start Ollama before running this script"
        exit 1
    fi
    print_success "Ollama is running"
}

# Function to create output directories
create_directories() {
    print_status "Creating output directories..."
    mkdir -p outputs/01_esg_testset
    mkdir -p outputs/02_rag_responses
    mkdir -p outputs/03_evaluation_results
    print_success "Output directories created"
}

# Function to run Step 0: Document Selection
run_step0() {
    print_status "Step 0: Selecting full ESG dataset (498 documents)..."
    echo "This will create the full dataset directory from the corpus..."
    
    if python code/00_select_documents.py; then
        print_success "Step 0 completed successfully"
        
        # Check if full dataset directory was created
        if [[ -d "data/documents" ]]; then
            doc_count=$(ls data/documents/*.md 2>/dev/null | wc -l)
            print_success "Full dataset created with $doc_count documents"
            echo "  - Dataset directory: data/documents/"
            echo "  - Selection info: outputs/document_selection_info.csv"
        else
            print_error "Full dataset directory not found"
            exit 1
        fi
    else
        print_error "Step 0 failed"
        exit 1
    fi
}

# Function to run Step 1: Generate Test Data
run_step1() {
    print_status "Step 1: Generating test data from full ESG dataset (498 documents)..."
    echo "This may take 30-60 minutes depending on your system..."
    
    if python code/01_generate_testset.py; then
        print_success "Step 1 completed successfully"
        
        # Check if output files were created
        if [[ -f "outputs/01_esg_testset/01_esg_knowledge_graph.pkl" && -f "outputs/01_esg_testset/01_esg_testset.csv" ]]; then
            print_success "Test data files generated"
            echo "  - Knowledge graph: outputs/01_esg_testset/01_esg_knowledge_graph.pkl"
            echo "  - Test set: outputs/01_esg_testset/01_esg_testset.csv"
        else
            print_error "Output files not found"
            exit 1
        fi
    else
        print_error "Step 1 failed"
        exit 1
    fi
}

# Function to run Step 2a: Compute and save document embeddings
run_step2a() {
    print_status "Step 2a: Computing document embeddings (once, saved to disk)..."
    echo "Embeds all corpus chunks with qwen3-embedding:4b, saves to outputs/01_esg_testset/doc_embeddings.npy"
    echo "Run once after Step 1 — Step 2 loads from file instead of recomputing."
    echo ""

    if python code/02a_compute_embeddings.py; then
        print_success "Step 2a completed successfully"

        if [[ -f "outputs/01_esg_testset/doc_embeddings.npy" ]]; then
            print_success "Embeddings saved: outputs/01_esg_testset/doc_embeddings.npy"
        else
            print_error "Embeddings file not found"
            exit 1
        fi
    else
        print_error "Step 2a failed"
        exit 1
    fi
}

# Function to run Step 2: RAG Evaluation
run_step2() {
    print_status "Step 2: Running RAG evaluation on 5 models with full dataset (498 documents)..."
    echo "This will evaluate the following models:"
    echo "  - Qwen Series: qwen3:4b-instruct (medium representative)"
    echo "  - Gemma Series: gemma3:4b (efficient and latest generation)"
    echo "  - Ministral Series: ministral-3:8b (European representative)"
    echo "  - Nemotron Series: nemotron-3-nano:4b (NVIDIA optimized)"
    echo "  - GLM Series: glm-4.7-flash:q4_K_M (quantized efficiency)"
    echo ""
    echo "Estimated time: 4-6 hours total (45 minutes-1.5 hours per model)"
    echo "Press Ctrl+C to interrupt and resume later"
    echo ""
    
    if python code/02_run_rag_evaluation.py; then
        print_success "Step 2 completed successfully"
        
        # Check if response files were created
        response_count=$(ls outputs/02_rag_responses/*.json 2>/dev/null | wc -l)
        if [[ $response_count -gt 0 ]]; then
            print_success "RAG responses generated for $response_count models"
            echo "Response files saved in: outputs/02_rag_responses/"
        else
            print_warning "No response files found - evaluation may have failed"
        fi
    else
        print_error "Step 2 failed"
        exit 1
    fi
}

# Function to run Step 3: Analyze Results
run_step3() {
    print_status "Step 3: Analyzing results with RAGAS metrics..."
    
    if python code/03_analyze_results.py; then
        print_success "Step 3 completed successfully"
        
        # Check if evaluation files were created
        eval_count=$(ls outputs/03_evaluation_results/*.csv 2>/dev/null | wc -l)
        if [[ $eval_count -gt 0 ]]; then
            print_success "Evaluation results generated"
            echo "Results saved in: outputs/03_evaluation_results/"
            echo "  - Summary CSV files"
            echo "  - Detailed JSON files"
        else
            print_warning "No evaluation files found"
        fi
    else
        print_error "Step 3 failed"
        exit 1
    fi
}

# Function to show experiment summary
show_summary() {
    print_status "Experiment Summary:"
    echo "  - Dataset: 498 ESG documents (2010-2024) - FULL CORPUS"
    echo "  - Models: 5 models across 5 families (4B-8B range)"
    echo "  - Selection: qwen3:4b-instruct, gemma3:4b, ministral-3:8b, nemotron-3-nano:4b, glm-4.7-flash:q4_K_M"
    echo "  - Metrics: 6 RAGAS evaluation metrics"
    echo "  - Output: Comprehensive results for academic publication"
    echo ""
    echo "Output files:"
    echo "  - outputs/01_esg_testset/ - Test data and knowledge graph"
    echo "  - outputs/02_rag_responses/ - RAG responses for each model"
    echo "  - outputs/03_evaluation_results/ - Evaluation results and summaries"
}

# Main execution
main() {
    echo "========================================"
    echo "ESG RAG Experiment Runner"
    echo "========================================"
    echo ""
    
    # Check prerequisites
    check_venv
    check_ollama
    create_directories
    
    # Show experiment summary
    show_summary
    echo ""
    
    # Ask user to confirm
    read -p "Do you want to proceed with the full experiment? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Experiment cancelled by user"
        exit 0
    fi
    
    # Record start time
    start_time=$(date +%s)
    echo "Experiment started at: $(date)"
    echo ""
    
    # Run all steps
    run_step0
    echo ""
    run_step1
    echo ""
    run_step2a
    echo ""
    run_step2
    echo ""
    run_step3
    
    # Calculate elapsed time
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))
    
    echo ""
    echo "========================================"
    print_success "Experiment completed successfully!"
    echo "========================================"
    echo "Total time: ${hours}h ${minutes}m"
    echo "Completed at: $(date)"
    echo ""
    echo "Results are available in the outputs/ directory"
    echo "Check outputs/03_evaluation_results/ for the summary CSV files"
}

# Handle command line arguments
case "${1:-}" in
    "step0")
        check_venv
        run_step0
        ;;
    "step1")
        check_venv
        check_ollama
        create_directories
        run_step1
        ;;
    "step2a")
        check_venv
        check_ollama
        create_directories
        run_step2a
        ;;
    "step2")
        check_venv
        check_ollama
        run_step2
        ;;
    "step3")
        check_venv
        run_step3
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [step0|step1|step2|step3|help]"
        echo ""
        echo "Commands:"
        echo "  step0  - Run only Step 0: Document selection (creates full dataset)"
        echo "  step1  - Run only Step 1: Generate test data"
        echo "  step2a - Run only Step 2a: Compute and save document embeddings (run once after step1)"
        echo "  step2  - Run only Step 2: RAG evaluation (requires step2a done first)"
        echo "  step3  - Run only Step 3: Analyze results"
        echo "  help   - Show this help message"
        echo ""
        echo "No arguments: Run the complete experiment (all steps)"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
