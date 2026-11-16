
import tkinter as tk
from tkinter import messagebox
import threading
import chess

from play import load_model, predict_moves


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

LIGHT_SQUARE = "#EEEED2"
DARK_SQUARE = "#769656"
SELECTED_COLOR = "#F6F669"
LAST_MOVE_COLOR = "#CDD26A"
LEGAL_MOVE_COLOR = "#526F3B"

BACKGROUND = "#20252B"
PANEL_BACKGROUND = "#292F36"
TEXT_COLOR = "#F2F2F2"
ACCENT_COLOR = "#81B64C"

PIECES = {
    chess.WHITE: {
        chess.PAWN: "♙",
        chess.KNIGHT: "♘",
        chess.BISHOP: "♗",
        chess.ROOK: "♖",
        chess.QUEEN: "♕",
        chess.KING: "♔",
    },
    chess.BLACK: {
        chess.PAWN: "♟",
        chess.KNIGHT: "♞",
        chess.BISHOP: "♝",
        chess.ROOK: "♜",
        chess.QUEEN: "♛",
        chess.KING: "♚",
    }
}


# ============================================================
# LOAD NEURAL NETWORK
# ============================================================

print("Initializing chess application...")

model, device = load_model()

print("Neural network ready!")


# ============================================================
# MAIN APPLICATION
# ============================================================

class ChessGUI:

    def __init__(self, root):

        self.root = root

        self.root.title("My Chess AI")
        self.root.geometry("1000x720")
        self.root.minsize(900, 680)
        self.root.configure(bg=BACKGROUND)

        self.board = chess.Board()

        self.selected_square = None
        self.legal_destinations = []

        self.ai_thinking = False
        self.game_id = 0

        self.move_history = []

        self.create_interface()
        self.draw_board()
        self.update_status()

    # ========================================================
    # CREATE INTERFACE
    # ========================================================

    def create_interface(self):

        main = tk.Frame(
            self.root,
            bg=BACKGROUND,
            padx=20,
            pady=15
        )

        main.pack(fill="both", expand=True)

        title = tk.Label(
            main,
            text="CHESS AI",
            font=("Segoe UI", 25, "bold"),
            fg=TEXT_COLOR,
            bg=BACKGROUND
        )

        title.pack(anchor="w", pady=(0, 15))

        content = tk.Frame(
            main,
            bg=BACKGROUND
        )

        content.pack(fill="both", expand=True)

        # Chessboard
        self.canvas = tk.Canvas(
            content,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            bg=BACKGROUND,
            highlightthickness=0
        )

        self.canvas.pack(
            side="left",
            padx=(0, 20)
        )

        self.canvas.bind(
            "<Button-1>",
            self.on_board_click
        )

        # Right panel
        panel = tk.Frame(
            content,
            bg=PANEL_BACKGROUND,
            width=280,
            padx=15,
            pady=15
        )

        panel.pack(
            side="right",
            fill="both",
            expand=True
        )

        panel.pack_propagate(False)

        # Player information
        tk.Label(
            panel,
            text="YOU  ·  WHITE",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT_COLOR,
            bg=PANEL_BACKGROUND
        ).pack(anchor="w")

        tk.Label(
            panel,
            text="VS",
            font=("Segoe UI", 11),
            fg="#999999",
            bg=PANEL_BACKGROUND
        ).pack(anchor="w", pady=5)

        tk.Label(
            panel,
            text="CHESS AI  ·  BLACK",
            font=("Segoe UI", 13, "bold"),
            fg=ACCENT_COLOR,
            bg=PANEL_BACKGROUND
        ).pack(anchor="w")

        # Status
        tk.Label(
            panel,
            text="GAME STATUS",
            font=("Segoe UI", 10, "bold"),
            fg="#AAAAAA",
            bg=PANEL_BACKGROUND
        ).pack(anchor="w", pady=(25, 5))

        self.status_label = tk.Label(
            panel,
            text="Your turn",
            font=("Segoe UI", 12, "bold"),
            fg=ACCENT_COLOR,
            bg=PANEL_BACKGROUND,
            wraplength=240
        )

        self.status_label.pack(anchor="w")

        # Thinking indicator
        self.thinking_label = tk.Label(
            panel,
            text="",
            font=("Segoe UI", 10),
            fg="#AAAAAA",
            bg=PANEL_BACKGROUND
        )

        self.thinking_label.pack(anchor="w", pady=5)

        # Move history
        tk.Label(
            panel,
            text="MOVE HISTORY",
            font=("Segoe UI", 10, "bold"),
            fg="#AAAAAA",
            bg=PANEL_BACKGROUND
        ).pack(anchor="w", pady=(20, 5))

        history_frame = tk.Frame(
            panel,
            bg=PANEL_BACKGROUND
        )

        history_frame.pack(
            fill="both",
            expand=True
        )

        self.history_text = tk.Text(
            history_frame,
            height=15,
            bg="#1C2127",
            fg=TEXT_COLOR,
            font=("Consolas", 11),
            relief="flat",
            state="disabled",
            wrap="word",
            padx=10,
            pady=10
        )

        self.history_text.pack(
            fill="both",
            expand=True
        )

        # Buttons
        button_frame = tk.Frame(
            panel,
            bg=PANEL_BACKGROUND
        )

        button_frame.pack(
            fill="x",
            pady=(15, 0)
        )

        tk.Button(
            button_frame,
            text="NEW GAME",
            command=self.new_game,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground="#6E9F3E",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=10,
            pady=10,
            cursor="hand2"
        ).pack(fill="x", pady=4)

        tk.Button(
            button_frame,
            text="UNDO LAST TURN",
            command=self.undo_move,
            bg="#444C55",
            fg="white",
            activebackground="#59636E",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=10,
            pady=10,
            cursor="hand2"
        ).pack(fill="x", pady=4)

    # ========================================================
    # DRAW CHESSBOARD
    # ========================================================

    def draw_board(self):

        self.canvas.delete("all")

        last_move = (
            self.board.peek()
            if self.board.move_stack
            else None
        )

        # Draw squares
        for row in range(8):

            for col in range(8):

                square = chess.square(
                    col,
                    7 - row
                )

                x1 = col * SQUARE_SIZE
                y1 = row * SQUARE_SIZE

                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE

                is_light = (row + col) % 2 == 0

                color = (
                    LIGHT_SQUARE
                    if is_light
                    else DARK_SQUARE
                )

                if last_move is not None:

                    if square in (
                        last_move.from_square,
                        last_move.to_square
                    ):
                        color = LAST_MOVE_COLOR

                if square == self.selected_square:
                    color = SELECTED_COLOR

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=color
                )

                # Coordinates
                if col == 0:

                    self.canvas.create_text(
                        x1 + 7,
                        y1 + 7,
                        text=str(8 - row),
                        anchor="nw",
                        fill=DARK_SQUARE if is_light else LIGHT_SQUARE,
                        font=("Segoe UI", 9, "bold")
                    )

                if row == 7:

                    self.canvas.create_text(
                        x2 - 7,
                        y2 - 5,
                        text=chess.FILE_NAMES[col],
                        anchor="se",
                        fill=DARK_SQUARE if is_light else LIGHT_SQUARE,
                        font=("Segoe UI", 9, "bold")
                    )

                # Legal move indicators
                if square in self.legal_destinations:

                    piece = self.board.piece_at(square)

                    center_x = x1 + SQUARE_SIZE // 2
                    center_y = y1 + SQUARE_SIZE // 2

                    if piece is None:

                        radius = 10

                        self.canvas.create_oval(
                            center_x - radius,
                            center_y - radius,
                            center_x + radius,
                            center_y + radius,
                            fill=LEGAL_MOVE_COLOR,
                            outline=""
                        )

                    else:

                        self.canvas.create_rectangle(
                            x1 + 3,
                            y1 + 3,
                            x2 - 3,
                            y2 - 3,
                            outline=LEGAL_MOVE_COLOR,
                            width=5
                        )

                # Draw piece
                piece = self.board.piece_at(square)

                if piece:

                    symbol = PIECES[
                        piece.color
                    ][
                        piece.piece_type
                    ]

                    self.canvas.create_text(
                        x1 + SQUARE_SIZE // 2,
                        y1 + SQUARE_SIZE // 2,
                        text=symbol,
                        font=("Segoe UI Symbol", 46),
                        fill=(
                            "#FFFFFF"
                            if piece.color == chess.WHITE
                            else "#151515"
                        ),
                        tags="piece"
                    )

    # ========================================================
    # BOARD CLICK HANDLING
    # ========================================================

    def on_board_click(self, event):

        if self.ai_thinking or self.board.is_game_over():
            return

        # Human plays White
        if self.board.turn != chess.WHITE:
            return

        col = event.x // SQUARE_SIZE
        row = event.y // SQUARE_SIZE

        if not (0 <= row < 8 and 0 <= col < 8):
            return

        square = chess.square(
            col,
            7 - row
        )

        piece = self.board.piece_at(square)

        # First click: select a white piece
        if self.selected_square is None:

            if piece and piece.color == chess.WHITE:
                self.select_square(square)

            return

        # Clicking the same square deselects it
        if square == self.selected_square:

            self.selected_square = None
            self.legal_destinations = []

            self.draw_board()
            return

        # Find legal move to clicked square
        possible_moves = [
            move
            for move in self.board.legal_moves
            if move.from_square == self.selected_square
            and move.to_square == square
        ]

        if possible_moves:

            # Automatically promote to Queen
            promotion_move = next(
                (
                    move for move in possible_moves
                    if move.promotion == chess.QUEEN
                ),
                possible_moves[0]
            )

            self.make_human_move(promotion_move)
            return

        # Select another white piece
        if piece and piece.color == chess.WHITE:

            self.select_square(square)
            return

        # Invalid destination
        self.selected_square = None
        self.legal_destinations = []

        self.draw_board()

    def select_square(self, square):

        self.selected_square = square

        self.legal_destinations = [
            move.to_square
            for move in self.board.legal_moves
            if move.from_square == square
        ]

        self.draw_board()

    # ========================================================
    # MAKE HUMAN MOVE
    # ========================================================

    def make_human_move(self, move):

        san = self.board.san(move)

        self.board.push(move)

        self.move_history.append(
            ("White", san)
        )

        self.selected_square = None
        self.legal_destinations = []

        self.update_history()
        self.draw_board()
        self.update_status()

        if not self.board.is_game_over():
            self.start_ai_move()

    # ========================================================
    # AI MOVE
    # ========================================================

    def start_ai_move(self):

        self.ai_thinking = True

        self.status_label.config(
            text="AI is thinking...",
            fg=ACCENT_COLOR
        )

        self.thinking_label.config(
            text="Calculating its next move..."
        )

        self.draw_board()

        # Copy position so the background thread doesn't
        # access a board being modified by the GUI.
        position = self.board.copy(stack=False)

        current_game_id = self.game_id

        def calculate():

            try:

                predictions = predict_moves(
                    model,
                    position,
                    device
                )

                move = (
                    predictions[0]["move"]
                    if predictions
                    else None
                )

                error = None

            except Exception as exception:

                move = None
                error = str(exception)

            self.root.after(
                0,
                lambda: self.finish_ai_move(
                    move,
                    error,
                    current_game_id
                )
            )

        threading.Thread(
            target=calculate,
            daemon=True
        ).start()

    def finish_ai_move(
        self,
        move,
        error,
        current_game_id
    ):

        # Ignore an old AI result if a new game was started
        if current_game_id != self.game_id:
            return

        self.ai_thinking = False

        self.thinking_label.config(text="")

        if error:

            messagebox.showerror(
                "AI Error",
                error
            )

            self.update_status()
            return

        if move is None or move not in self.board.legal_moves:

            messagebox.showerror(
                "AI Error",
                "The AI did not return a legal move."
            )

            self.update_status()
            return

        san = self.board.san(move)

        self.board.push(move)

        self.move_history.append(
            ("Black", san)
        )

        self.update_history()
        self.draw_board()
        self.update_status()

    # ========================================================
    # MOVE HISTORY
    # ========================================================

    def update_history(self):

        self.history_text.config(
            state="normal"
        )

        self.history_text.delete(
            "1.0",
            tk.END
        )

        for index in range(
            0,
            len(self.move_history),
            2
        ):

            white_move = self.move_history[index][1]

            black_move = (
                self.move_history[index + 1][1]
                if index + 1 < len(self.move_history)
                else "..."
            )

            move_number = index // 2 + 1

            self.history_text.insert(
                tk.END,
                f"{move_number}. {white_move:<9} {black_move}\n"
            )

        self.history_text.config(
            state="disabled"
        )

        self.history_text.see(tk.END)

def undo_move(self):
    if self.ai_thinking or not self.board.move_stack:
        return

    self.game_id += 1

    # Undo both moves if the AI has already responded.
    plies_to_undo = (
        2 if self.board.turn == chess.WHITE
        and len(self.board.move_stack) >= 2
        else 1
    )

    for _ in range(plies_to_undo):
        self.board.pop()
        if self.move_history:
            self.move_history.pop()

    self.selected_square = None
    self.legal_destinations = []

    self.update_history()
    self.draw_board()
    self.update_status()

    # ========================================================
    # GAME STATUS
    # ========================================================

    def update_status(self):

        if self.board.is_checkmate():

            winner = (
                "White wins!"
                if self.board.turn == chess.BLACK
                else "Black wins!"
            )

            self.status_label.config(
                text=f"CHECKMATE — {winner}",
                fg="#E5B85C"
            )

            return

        if self.board.is_stalemate():

            self.status_label.config(
                text="Draw — Stalemate",
                fg="#E5B85C"
            )

            return

        if self.board.is_insufficient_material():

            self.status_label.config(
                text="Draw — Insufficient material",
                fg="#E5B85C"
            )

            return

        if self.board.is_game_over():

            self.status_label.config(
                text=f"Game over — {self.board.result()}",
                fg="#E5B85C"
            )

            return

        if self.ai_thinking:

            self.status_label.config(
                text="AI is thinking...",
                fg=ACCENT_COLOR
            )

            return

        if self.board.is_check():

            self.status_label.config(
                text="CHECK!",
                fg="#FF7777"
            )

            return

        if self.board.turn == chess.WHITE:

            self.status_label.config(
                text="Your turn — White",
                fg=ACCENT_COLOR
            )

        else:

            self.status_label.config(
                text="AI's turn — Black",
                fg=ACCENT_COLOR
            )

    # ========================================================
    # NEW GAME
    # ========================================================

    def new_game(self):

        self.game_id += 1

        self.board.reset()

        self.selected_square = None
        self.legal_destinations = []

        self.ai_thinking = False
        self.move_history = []

        self.update_history()
        self.draw_board()
        self.update_status()

    # ========================================================
    # UNDO LAST TURN
    # ========================================================

def undo_move(self):

    if self.ai_thinking or not self.board.move_stack:
        return

    self.game_id += 1

    # If Black just moved, undo both Black's and White's moves.
    # If White just moved, undo only White's move.
    plies_to_undo = (
        2 if self.board.turn == chess.WHITE
        and len(self.board.move_stack) >= 2
        else 1
    )

    for _ in range(plies_to_undo):
        self.board.pop()
        if self.move_history:
            self.move_history.pop()

    self.selected_square = None
    self.legal_destinations = []

    self.update_history()
    self.draw_board()
    self.update_status()


# ============================================================
# START APPLICATION
# ============================================================
if __name__ == "__main__":

    root = tk.Tk()

    app = ChessGUI(root)

    root.mainloop()
