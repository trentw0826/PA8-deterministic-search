#!/usr/bin/env python3
"""
Test script to run the algorithm comparison and generate the plot
"""

import sys
import argparse

# Import the plotting module
from plotting import test_algorithms_and_plot
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PUZZLE_FILE

def main():
    """Main function that handles command line arguments"""
    parser = argparse.ArgumentParser(description="Test and compare 8-puzzle search algorithms")
    parser.add_argument('filename', nargs='?', default=DEFAULT_PUZZLE_FILE, 
                       help=f'Puzzle file to test (default: {DEFAULT_PUZZLE_FILE})')
    parser.add_argument('-l', '--list-files', action='store_true',
                       help='List available puzzle files')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                       help=f'Input directory for puzzle files (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for generated plots (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # List available files if requested
    if args.list_files:
        import os
        input_path = args.input_dir
        if os.path.exists(input_path):
            puzzle_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
            print(f"Available puzzle files in {input_path}:")
            for file in sorted(puzzle_files):
                print(f"  {file}")
        else:
            print(f"Input directory '{input_path}' not found.")
        return
    
    # Validate file exists
    import os
    puzzle_path = os.path.join(args.input_dir, args.filename)
    if not os.path.exists(puzzle_path):
        print(f"Error: File '{puzzle_path}' not found.")
        print("Use --list-files to see available puzzle files.")
        sys.exit(1)
    
    # Run the test function
    print(f"Running algorithm comparison test on {args.filename}...")
    print("=" * 60)
    results = test_algorithms_and_plot(args.filename, args.input_dir, args.output_dir)
    print("\nTest completed!")

if __name__ == '__main__':
    main()