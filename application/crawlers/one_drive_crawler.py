from onedrivesdk import HttpProvider, Client, AuthProvider
from onedrivesdk.helpers import GetAuthCodeServer

class OneDriveCrawler:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize'
        self.scopes = ['wl.signin', 'wl.offline_access', 'onedrive.readwrite']
        self.auth_provider = None
        self.http_provider = HttpProvider()
        self.client = None

    def authenticate(self):
        # Configura o provedor de autenticação
        self.auth_provider = AuthProvider(
            self.auth_url,
            self.client_id,
            self.scopes,
            self.redirect_uri,
            self.client_secret
        )

        # Obtém o código de autorização
        auth_code_url = self.auth_provider.get_auth_url()
        code_receiver = GetAuthCodeServer(auth_code_url, self.redirect_uri)
        code = code_receiver.get_auth_code()

        # Autentica o cliente
        self.client = Client(self.auth_provider, self.http_provider)
        self.client.auth_provider.authenticate(code, self.redirect_uri, self.client_secret)

    def list_files_in_shared_folder(self, shared_folder_id):
        if not self.client:
            raise Exception('Cliente não autenticado. Chame o método authenticate() primeiro.')

        shared_drive = self.client.drive(id=shared_folder_id)

        items = shared_drive.root.children.get()
        for item in items:
            print(item.name)

# Exemplo de uso da classe OneDriveCrawler
if __name__ == "__main__":
    # Configurações de autenticação
    client_id = 'seu_client_id'
    client_secret = 'seu_client_secret'
    redirect_uri = 'http://localhost'

    # Instancia a classe OneDriveCrawler
    crawler = OneDriveCrawler(client_id, client_secret, redirect_uri)

    # Autentica o cliente
    crawler.authenticate()

    shared_folder_id = "<>"
    crawler.list_files_in_shared_folder(shared_folder_id)
