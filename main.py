from src import (
    create_directories,
    print_header,
    prepare_data,
    vectorize_data,
    train_all_models,
    visualize_results,
    print_summary,
    save_all_reports,
    print_all_reports
)


def main():
    """Main execution function"""
      
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Load and prepare data
    print_header("Data Preparation")
    X_train, X_test, y_train, y_test, label_to_id, id_to_label = prepare_data()
    
    # Step 3: Vectorize data
    print_header("Data Vectorization")
    vectorized_data = vectorize_data(X_train, X_test)
    
    # Step 4: Train models
    print_header("Model Training")
    results = train_all_models(vectorized_data, y_train, y_test, id_to_label)
    
    # Step 5: Print summary
    print_summary(results)
    
    # Step 6: Print detailed reports 
    print_all_reports(results)
    
    # Step 7: Save reports to files
    print_header("Saving Reports")
    save_all_reports(results)
    
    # Step 8: Visualize results
    print_header("Visualization")
    visualize_results(results, y_test, id_to_label)
    

if __name__ == "__main__":
    main()