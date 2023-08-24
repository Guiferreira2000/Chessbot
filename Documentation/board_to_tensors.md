<h1>Chapter 2: </h1> <h2>Converting Board Positions to Tensors</h2>

The board positions extracted from the PGN file are in FEN (Forsyth-Edwards Notation) format. FEN is a compact string representation of a chess board position. However, neural networks can't work directly with FEN strings. Instead, they need numerical data, typically in the form of tensors.

The function fen_to_tensor converts a FEN string to a tensor. Here's a breakdown:

    The FEN string is split to only take the piece placement data. A typical FEN string looks like this: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1". The piece placement data is the first part: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR".

    The piece placement data is further split into rows (ranks in chess terminology). Each row is processed character by character:

        If the character is a digit, it represents empty squares. The column index is incremented by that number.

        If the character is a letter, it represents a chess piece. The tensor is updated at the current row and column index with a one-hot encoded vector representing the piece. The function piece_to_index maps each piece to a unique index in the range [0, 11].

The resulting tensor has a shape of (8, 8, 12). The first two dimensions represent the 8x8 chess board, and the third dimension represents the 12 possible pieces (6 for each color: Pawn, Knight, Bishop, Rook, Queen, King).

This tensor format allows the neural network to recognize patterns in the spatial arrangement of pieces on the board.