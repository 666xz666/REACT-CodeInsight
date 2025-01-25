from .code2ast import code2ast
from .get_ast_diff import get_ast_diff


def extract_ast_seq(neg_code, pos_code):
    neg_ast = code2ast(neg_code)
    pos_ast = code2ast(pos_code)
    ast_diff = get_ast_diff(neg_ast, pos_ast)

    return ast_diff


if __name__ == '__main__':
    neg_code = """
        public void stop() throws Exception {
        if (closed.compareAndSet(false, true)) {
            closing.set(true);
            ServiceStopper stopper = new ServiceStopper();
            try {
                doStop(stopper);
            }
            catch (Exception e) {
                stopper.onException(this, e);
            }
            if (runner != null) {
                runner.join();
                runner = null;
            }
            closed.set(true);
            started.set(false);
            closing.set(false);
            stopper.throwFirstException();
        }
    }
    """
    pos_code = """
        private boolean joinOnStop = true;
    public void stop() throws Exception {
        if (closed.compareAndSet(false, true)) {
            closing.set(true);
            ServiceStopper stopper = new ServiceStopper();
            try {
                doStop(stopper);
            }
            catch (Exception e) {
                stopper.onException(this, e);
            }
            if (runner != null && joinOnStop) {
                runner.join();
                runner = null;
            }
            closed.set(true);
            started.set(false);
            closing.set(false);
            stopper.throwFirstException();
        }
    }
    public boolean isJoinOnStop() {
        return joinOnStop;
    }
    public void setJoinOnStop(boolean joinOnStop) {
        this.joinOnStop = joinOnStop;
    }
    """

    ast_diff = extract_ast_seq(neg_code, pos_code)
    print(ast_diff)
