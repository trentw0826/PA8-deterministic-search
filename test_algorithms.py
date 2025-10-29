#!/usr/bin/env python3
"""
Test script to run the algorithm comparison and generate the plot
"""
import sys
import argparse

# Import our eight.py module
from eight import test_algorithms_and_plot

def main():
    """Main function that handles command line arguments"""
    parser = argparse.ArgumentParser(description="Test and compare 8-puzzle search algorithms")
    parser.add_argument('filename', nargs='?', default='easy.txt', 
                       help='Puzzle file to test (default: easy.txt)')
    parser.add_argument('-l', '--list-files', action='store_true',
                       help='List available puzzle files')
    
    args = parser.parse_args()
    
    # List available files if requested
    if args.list_files:
        import os
        puzzle_files = [f for f in os.listdir('.') if f.endswith('.txt')]
        print("Available puzzle files:")
        for file in sorted(puzzle_files):
            print(f"  {file}")
        return
    
    # Validate file exists
    import os
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.")
        print("Use --list-files to see available puzzle files.")
        sys.exit(1)
    
    # Run the test function
    print(f"Running algorithm comparison test on {args.filename}...")
    print("=" * 60)
    results = test_algorithms_and_plot(args.filename)
    print("\nTest completed!")

if __name__ == '__main__':
    main()