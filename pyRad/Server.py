"""
Wrapper for a remote server performing RT-related tasks.

Copyright Marc-Andre Renaud, 2017
"""
import os
import json
import paramiko


class Server(object):
    """
    Server from which dose simulations are performed.

    Attributes:

    """

    def __init__(self, attrs):
        """
        Constructor.

        :param name: Server name
        :param username: Username of account on server.
        :param host: IP or hostname of server.
        :param port: Port to use when connecting to server. (default = 22)
        :param sim_paths: Absolute paths to relevant installed software.
        """
        for k, v in attrs.items():
            setattr(self, k, v)

    def __del__(self):
        """
        Destructor.

        Ensures connections are closed when class is destroyed.
        """
        try:
            self.__ssh.close()
            self.__sftp.close()
        except AttributeError:
            pass

    def _open_connection(self):
        try:
            return (self.__ssh, self.__sftp)
        except AttributeError:
            self.__ssh = paramiko.SSHClient()
            self.__ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.__ssh.connect(hostname=self.host,
                               port=self.port,
                               username=self.username)

            self.__sftp = self.__ssh.open_sftp()

            return (self.__ssh, self.__sftp)

    def add_key(self, password):
        """
        Add ssh key to server for passwordless login for username.

        :param password: password of account on server.
        """
        with open(os.path.expanduser('~/.ssh/id_rsa.pub')) as pubkey:
            key = pubkey.read()

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host, port=self.port, username=self.username, password=password)
        client.exec_command('mkdir -p ~/.ssh/')
        client.exec_command('echo "%s" >> ~/.ssh/authorized_keys' % key)
        client.exec_command('chmod 644 ~/.ssh/authorized_keys')
        client.exec_command('chmod 700 ~/.ssh/')

    def check_connection(self):
        """Check if connection to server can be made."""
        self._open_connection()
        return True

    def get_path(self, program):
        """
        Return the path to a specific program installed on the server.

        :param program: The name of the program.
        """
        try:
            return self.sim_paths[program]
        except KeyError:
            raise Exception("Could not find path to program: {} in server: {}".format(program, self.name))

    def exec_command(self, command):
        """
        Execute a command on the server.

        :param command: Command to execute.
        """
        ssh, sftp = self._open_connection()
        return ssh.exec_command(command)

    def exec_shell_command(self, command):
        """
        Execute multiple commands in the same session.

        :param command: Commands to execute.
        """
        ssh, sftp = self._open_connection()

        channel = ssh.invoke_shell()
        stdin = channel.makefile("wb")
        stdout = channel.makefile("rb")
        stderr = channel.makefile("rb")

        stdin.write(command)
        response = stdout.read()

        stdin.close()
        stdout.close()
        stderr.close()
        channel.close()

        return response

    def put_files(self, file_list, directory, make_dir=False, delete_original=True):
        """
        Transfer files to server.

        :param file_list: List of files to transfer.
        :param directory: Remote directory for transfer.
        :param make_dir: If true, make directory if it does not exist.
        :param delete_original: If true, delete the local file.
        """
        ssh, sftp = self._open_connection()

        if make_dir:
            sftp.mkdir(directory)
        sftp.chdir(directory)

        for file_to_send in file_list:
            sftp.put(file_to_send, os.path.basename(file_to_send))
            if delete_original:
                os.remove(file_to_send)

    def move_file(self, from_file, to_folder, make_dir=False):
        """
        Move file to a new folder on remote server.

        :param from_file: Original file path.
        :param to_folder: Destination folder of the file.
        :param make_dir: If true, create directory if it does not exist.
        """
        ssh, sftp = self._open_connection()

        if make_dir:
            sftp.mkdir(to_folder)

        return sftp.mv(from_file, to_folder)

    def get_folder_size(self, directory):
        """
        Find the total size of all files in a directory.

        :param directory: Directory to find file sizes.
        """
        size_command = 'du %s' % directory
        stdin, stdout, stderr = self.exec_command(size_command)
        output = stdout.read()
        try:
            folder_size = int(output.split()[0])
        except:
            folder_size = -1
        return folder_size

    def get_file_list(self, directory):
        """
        Get list of files in a directory.

        :param directory: Directory to find list of files.
        """
        ssh, sftp = self._open_connection()
        sftp.chdir(directory)

        try:
            sftp.stat("file_list.json")
            sftp.get("file_list.json", "file_list.json")

            with open("file_list.json", "r") as myfile:
                files_in_dir = json.load(myfile)
        except IOError:
            files_in_dir = sftp.listdir()

        return files_in_dir

    def stat(self, path):
        """
        Stat a file at a specific path.

        :param path: Path of file to stat.
        """
        ssh, sftp = self._open_connection()
        return sftp.stat(path)

    def get_file(self, remote_path, local_path):
        """
        Retrieve a remote file.

        :param remote_path: Path of file to retrieve.
        :param local_path: Local path of file after retrieval.
        """
        ssh, sftp = self._open_connection()
        return sftp.get(remote_path, local_path)

    def remove(self, path):
        """
        Remove a remote file.

        :param path: Path of file to remove.
        """
        ssh, sftp = self._open_connection()
        return sftp.remove(path)
